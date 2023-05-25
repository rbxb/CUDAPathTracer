#include "constants.h"
#include "pinhole_camera.h"
#include <float.h>
#include <assert.h>
#include "point.h"
#include "object.h"
#include "obj.cpp"
#include <vector>
#include "aabb.h"
#include "netpbm.cpp"
#include "intersect.cu"
#include "bounding_sphere.cpp"

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512

void init(bool print) {
    int device;
    HANDLE_CUDA_ERROR(cudaGetDevice(&device));

    struct cudaDeviceProp props;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&props, device));

    if (print) printf("%s Compute Capability %d.%d\n", props.name, props.major, props.minor);

    assert(sizeof(Point) == sizeof(float) * 3);
    assert(sizeof(Ray) == sizeof(float) * 6);
}

int main(int argc, char* argv[]) {
    init(true);

    // Create camera
    Camera* cam = new PinholeCamera();
    cam->setOutputFrame(IMAGE_WIDTH, IMAGE_HEIGHT);
    //cam->setView(Ray(Point(0.3f,1.5f,-20), Point(0,0,1)), 24.0f, 10.0f);

    Point cam_pos = Point(6,-2,-10);
    Point look_at = Point(0.5f,0.5f,0.5f);

    cam->setView(Ray(cam_pos, look_at - cam_pos), 60.0f, 10.0f);

    // Create initial rays
    int n;
    Ray* h_rays = cam->createRays(&n);

    // Load objects
    obj::ObjectReader reader = obj::ObjectReader();
    std::vector<IndexedObject> objects = reader.read("resources/cube-tex.obj");
    IndexedObject indexed_teapot = objects[0];
    ExpandedObject teapot = indexed_teapot.expand();

    // Load texture
    int tex_width, tex_height;
    netpbm::tupltype_t type;
    float* texture = netpbm::load_image("resources/texture.ppm", &tex_width, &tex_height, &type);

    // Create bounding boxes
    Point sphereCenter;
    float sphereRadius;
    boundingSphere(teapot.verts, teapot.n, sphereCenter, sphereRadius);
    
    // Create image and depth map
    float *image = (float*)calloc(IMAGE_WIDTH * IMAGE_HEIGHT * 3, sizeof(float));
    float *depth = (float*)calloc(IMAGE_WIDTH * IMAGE_HEIGHT, sizeof(float));

    // iterate over rays
    for (int i = 0; i < n; i++) {
        Ray ray = h_rays[i];
        if (raySphereIntersection(ray, sphereCenter, sphereRadius)) {
            float t, u, v;
            for (int j = 0; j < teapot.n / 3; j++) {
                if (intersectTriangle(ray, &teapot.verts[j*3], t, u, v) && t > depth[i]) {
                    float w = 1 - u - v;

                    Point A = teapot.tex_coords[j*3];
                    Point B = teapot.tex_coords[j*3+1];
                    Point C = teapot.tex_coords[j*3+2];

                    Point P = B * u + C * v + A * w;

                    int tex_x = int(P.x * tex_width);
                    int tex_y = int((1 - P.y) * tex_height);
                    int tex_i = tex_y * tex_width + tex_x;
                    memcpy(&image[i*3], &texture[tex_i*3], 3 * sizeof(float));

                    depth[i] = t;
                }
            }

            if (image[i * 3] == 0) {
                image[i * 3] = 0.2f;
            }
        }
    }

    // Save image
    netpbm::save_image("out/aabb.ppm", image, IMAGE_WIDTH, IMAGE_HEIGHT, netpbm::TUPLTYPE_RGB);
}
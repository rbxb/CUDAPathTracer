#include "constants.h"
#include "prefixsum.cu"
#include "object.h"
#include "geometrybox.h"
#include "pinhole_camera.h"
#include "obj.cpp"
#include "netpbm.cpp"
#include "intersect.cu"
#include <float.h>
#include "shared_unpack.cu"

#define IMAGE_WIDTH 1024
#define MAX_INITIAL_RAYS IMAGE_WIDTH * IMAGE_WIDTH
#define BLOCKSIZE 1024

#define LIGHT_DIR (normalize(Point(0.1f,1.0f,0.0f)))

__global__
void initialRaysKernel(Ray* rays, int n, Camera* cam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rays[i] = cam->createRay(i, 0);
}

#define TRIS_PER_TILE BLOCKSIZE
__global__
void renderKernel(Ray* rays, int numRays, Point* verts, int numTris, Point* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;

    Ray ray = rays[i];
    float d = FLT_MAX;
    Point normal;
    Point color = Point(0);

    for (int tr = 0; tr < numTris; tr++) {
        if (intersectTriangle(ray, &verts[tr * 3], d, normal)) {
            color = Point(1) * (dot(normal, LIGHT_DIR) + 0.2f);
        }
    }

    output[i] = min(color, Point(1));
}

int main(int argc, char* argv[]) {
    init(true);

    Camera* cam = new PinholeCamera(100, 90);
    cam->setOutputFrame(IMAGE_WIDTH, IMAGE_WIDTH, 1);
    cam->setOrientation(Ray(Point(0,0,0), Point(0,-0.2f,1)));
    Camera* d_cam = cam->upload();

    Ray* d_initialRays;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_initialRays, MAX_INITIAL_RAYS * sizeof(Ray)));
    initialRaysKernel<<<(MAX_INITIAL_RAYS - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(d_initialRays, MAX_INITIAL_RAYS, d_cam);

    Point* image = (Point*)malloc(IMAGE_WIDTH * IMAGE_WIDTH * sizeof(Point));
    Point* d_image;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_image, IMAGE_WIDTH * IMAGE_WIDTH * sizeof(Point)));

    obj::Object objObject = obj::read("resources/teapot.obj")[0];
    VerticesObject object = objObject.toVerticesObject();
    object.rotate(RodriguesRotation(Point(0,0,1), Point(1,0,0)));
    object.translate(Point(0,-4,10));
    GeometryBox* box = new GeometryBox();
    box->add(object);

    Point* d_boxBuffer;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_boxBuffer, box->bufferSize()));
    box->upload(d_boxBuffer);

    renderKernel<<<(MAX_INITIAL_RAYS - 1) / BLOCKSIZE + 1, BLOCKSIZE, TRIS_PER_TILE * 3 * sizeof(Point)>>>(d_initialRays, MAX_INITIAL_RAYS, d_boxBuffer, box->numTris(), d_image);
    HANDLE_CUDA_ERROR(cudaMemcpy(image, d_image, IMAGE_WIDTH * IMAGE_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    netpbm::save_image("./out/out.ppm", (float*)image, IMAGE_WIDTH, IMAGE_WIDTH, netpbm::TUPLTYPE_RGB);
    printf("Saved as ./out/out.ppm\n");
}
#include "pinhole_camera.h"
#include "constants.h"


int main(int argc, char* argv[]) {
    Camera* cam = new PinholeCamera(20, 40);
    cam->setOutputFrame(1920, 1080, 8);
    cam->setOrientation(Ray(Point(0,0,0), Point(0,1,0)));
    Camera* d_cam = cam->upload();

    return 0;
}
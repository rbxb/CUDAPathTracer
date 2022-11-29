#pragma once

#include "camera.h"
#include "constants.h"
#include "point.h"
#include "rotate.h"

#include <new>

class PinholeCamera : public Camera {
public:
    PinholeCamera(float fovWidth, float fovDegrees);
    void setOutputFrame(int width, int height, int density);
    void setOrientation(Ray orientation);
    Camera* upload();
    __device__ Ray createRay(int pixel, int k);

private:
    float fovw;
    float fovh;
    float d;
    int framew;
    int frameh;
    int density;
    Point origin;
    RodriguesRotation rotation;
};

PinholeCamera::PinholeCamera(float fovWidth, float fovDegrees) {
    fovw = fovWidth;
    float theta = fovDegrees * 0.01745329f; // convert degrees to radians
    d = fovw / tanf(theta * 0.5f);

    fovh = framew = frameh = fovw;
    origin = Point(0,0,0);
    rotation = RodriguesRotation();
}

void PinholeCamera::setOutputFrame(int width, int height, int density) {
    framew = width;
    frameh = height;
    this->density = density;
    fovh = ((float)height) / width * fovw;
}

void PinholeCamera::setOrientation(Ray orientation) {
    origin = orientation.origin;
    rotation = RodriguesRotation(Point(0,0,1), orientation.dir);
}

namespace __PinholeCamera_private_upload_kernel {
    __global__
    void kernel(PinholeCamera c, PinholeCamera* p) {
        p = new(p) PinholeCamera(c);
    }
};

Camera* PinholeCamera::upload() {
    PinholeCamera* p;
    HANDLE_CUDA_ERROR(cudaMalloc(&p, sizeof(PinholeCamera)));
    __PinholeCamera_private_upload_kernel::kernel<<<1,1>>>(*this, p);
    return p;
}

__device__
Ray PinholeCamera::createRay(int pixel, int k) {
    float u = (pixel % framew) / fovw - fovw * 0.5f;
    float v = (pixel / framew) / fovh - fovh * 0.5f;
    v *= -1; // flip y axis

    Point dir = rotation.rotate(Point(u,v,d));
    return Ray(origin, dir);
}
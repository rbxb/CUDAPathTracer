#pragma once

#include "point.h"

class Camera {
public:
    virtual void setOutputFrame(int width, int height, int density) = 0;
    virtual void setOrientation(Ray orientation) = 0;
    virtual Camera* upload() = 0;
    __device__ virtual Ray createRay(int pixel, int k) = 0;
};
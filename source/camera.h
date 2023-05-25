#pragma once

#include "point.h"
#include "rotate.h"

class Camera {
public:
    void setOutputFrame(int width, int height);
    void setView(Ray view, float fov, float d);
    virtual Ray* createRays(int* n) = 0;
    int width;
    int height;
    Point origin;
    RodriguesRotation rotation;
    float fov;
    float d;
};

void Camera::setOutputFrame(int width, int height) {
    this->width = width;
    this->height = height;
}

void Camera::setView(Ray view, float fov, float d) {
    this->fov = fov;
    this->d = d;
    this->origin = view.origin;
    this->rotation = RodriguesRotation(Point(0,0,1), normalize(view.dir));
}
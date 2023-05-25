#pragma once

#include "camera.h"
#include "constants.h"
#include "point.h"
#include "rotate.h"
#include <new>

class PinholeCamera : public Camera {
public:
    Ray* createRays(int* n);

private:
    Ray createRay(int y, int x, float world_w, float world_h);
};

Ray* PinholeCamera::createRays(int* n) {
    *n = width * height;
    Ray* a = (Ray*)malloc(sizeof(Ray) * *n);

    float theta = fov * 0.01745329f;
    float world_w = tanf(theta * 0.5f) * 2 * d;
    float world_h = world_w * height / width;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            a[y * width + x] = createRay(x, y, world_w, world_h);
        }
    }

    return a;
}

Ray PinholeCamera::createRay(int x, int y, float world_w, float world_h) {
    float u = (float(x) / width - 0.5f) * world_w;
    float v = (float(y) / height - 0.5f) * world_h * -1;

    Point dir = normalize(rotation.rotate(Point(u, v, d)));
    return Ray(origin, dir);
}
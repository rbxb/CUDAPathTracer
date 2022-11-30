#pragma once

#include "point.h"
#include "rotate.h"

struct VerticesObject {
    Point* verts;
    int n;

    __host__ void translate(Point t);
    __host__ void rotate(RodriguesRotation rotation);
};

__host__
void VerticesObject::translate(Point t) {
    for (int i = 0; i < n; i++) {
        verts[i] += t;
    }
}

__host__
void VerticesObject::rotate(RodriguesRotation rotation) {
    for (int i = 0; i < n; i++) {
        verts[i] = rotation.rotate(verts[i]);
    }
}
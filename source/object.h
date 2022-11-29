#pragma once

#include "point.h"
#include <cfloat>

struct VerticesObject {
    float* verts;
    int n;

    __host__ __device__ inline Point operator[](int index) {return Point(verts, index);};
};
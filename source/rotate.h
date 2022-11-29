#pragma once

#include "point.h"
#include <cmath>

// Rodrigues' rotation formula
// Finds the rotation needed to get from base to newBase, then applies that rotation to v

struct RodriguesRotation {
    __host__ __device__ RodriguesRotation() {axis=Point(); theta=0;};
    __host__ __device__ RodriguesRotation(const Point& base, const Point& newBase);
    __host__ __device__ Point rotate(const Point& v);

    Point axis;
    float theta;
};

__host__ __device__
RodriguesRotation::RodriguesRotation(const Point& base, const Point& newBase) {
    axis = normalize(cross(base, newBase));
    theta = acosf(dot(base, newBase) / (magnitude(base) * magnitude(newBase)));
}

__host__ __device__
Point RodriguesRotation::rotate(const Point& v) {
    return v * cosf(theta) + cross(axis, v) * sinf(theta) + axis * dot(axis, v) * (1 - cosf(theta));
}
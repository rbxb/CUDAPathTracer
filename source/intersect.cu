#pragma once

#include "point.h"

__device__
inline bool intersectTriangle(const Ray& ray, const Point* verts, float& old_d, Point& normal) {
    Point e1 = verts[1] - verts[0];
    Point e2 = verts[2] - verts[0];

    Point pvec = cross(ray.dir, e2);
    float det = dot(pvec, e1);
    if (det < SMALL_FLT && det > -SMALL_FLT) return false;

    float invDet = 1.0f / det;
    Point tvec = ray.origin - verts[0];
    float u = invDet * dot(tvec, pvec);
    if (u < 0.0f || u > 1.0f) return false;

    Point qvec = cross(tvec, e1);
    float v = invDet * dot(qvec, ray.dir);
    if (v < 0.0f || u+v > 1.0f) return false;

    float d = invDet * dot(e2, qvec);
    if (d < 0.0f || d > old_d) return false;

    old_d = d;
    normal = cross(e1, e2);
    return true;
}
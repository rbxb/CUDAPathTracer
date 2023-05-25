#pragma once

#include "point.h"

#define SMALL_FLT 0.000001f

__host__ __device__
inline bool intersectTriangle(const Ray& ray, const Point* verts, float& d, float& u, float& v) {
    Point e1 = verts[1] - verts[0];
    Point e2 = verts[2] - verts[0];

    Point pvec = cross(ray.dir, e2);
    float det = dot(pvec, e1);
    if (det < SMALL_FLT && det > -SMALL_FLT) return false;

    float invDet = 1.0f / det;
    Point tvec = ray.origin - verts[0];
    u = invDet * dot(tvec, pvec);
    if (u < 0.0f || u > 1.0f) return false;

    Point qvec = cross(tvec, e1);
    v = invDet * dot(qvec, ray.dir);
    if (v < 0.0f || u+v > 1.0f) return false;

    d = invDet * dot(e2, qvec);
    if (d < 0.0f) return false;

    return true;
}


/*__host__ __device__
inline bool intersectTriangle(const Ray& ray, const Point* verts, float& t, float& u, float& v) {
    Point N = cross(verts[1] - verts[0], verts[2] - verts[0]); // triangle normal
    float denom = dot(N, N);

    // Check if ray and plane are parallel
    float NdotDir = dot(N, ray.dir);
    if (fabs(NdotDir) < SMALL_FLT) return false;

    float d = dot(N, verts[1]);

    // Compute distance to intersection
    t = (dot(N, ray.origin) + d) / NdotDir;
    if (t < 0) return false;

    // Compute intersection point
    Point P = ray.origin + ray.dir * t;

    // Inside-Outside test
    Point C, edge, p;

    edge = verts[1] - verts[0];
    p = P - verts[0];
    C = cross(edge, p);
    if (dot(N, C) < 0) return false;

    edge = verts[2] - verts[1];
    p = P - verts[1];
    C = cross(edge, p);
    u = dot(N, C);
    if (u < 0) return false;

    edge = verts[0] - verts[2];
    p = P - verts[2];
    C = cross(edge, p);
    v = dot(N, C);
    if (v < 0) return false;

    u /= denom;
    v /= denom;

    return true;
}*/

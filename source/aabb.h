#include "point.h"
#include <float.h>
#include <algorithm>

struct Box {
    Point min;
    Point max;
};

Box aabb(Point* verts, int n) {
    Point min = verts[0];
    Point max = verts[0];

    for (int i = 1; i < n; i++) {
        Point v = verts[i];
        min.x = fmin(min.x, v.x);
        min.y = fmin(min.y, v.y);
        min.z = fmin(min.z, v.z);
        max.x = fmax(max.x, v.x);
        max.y = fmax(max.y, v.y);
        max.z = fmax(max.z, v.z);
    }

    return Box{min,max};
}

bool aabb_intersection(const Ray& ray, const Box& box)
{
    float tmin = 0, tmax = FLT_MAX;
    Point dir_inv = inverse(ray.dir);
    float t1, t2;

    t1 = (box.min.x - ray.origin.x) * dir_inv.x;
    t2 = (box.max.x - ray.origin.x) * dir_inv.x;
    tmin = fmax(tmin, fmin(t1, t2));
    tmax = fmin(tmax, fmax(t1, t2));

    t1 = (box.min.y - ray.origin.y) * dir_inv.y;
    t2 = (box.max.y - ray.origin.y) * dir_inv.y;
    tmin = fmax(tmin, fmin(t1, t2));
    tmax = fmin(tmax, fmax(t1, t2));

    t1 = (box.min.z - ray.origin.z) * dir_inv.z;
    t2 = (box.max.z - ray.origin.z) * dir_inv.z;
    tmin = fmax(tmin, fmin(t1, t2));
    tmax = fmin(tmax, fmax(t1, t2));

    return tmin < tmax;
}
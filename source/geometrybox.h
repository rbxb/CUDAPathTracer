#pragma once

#include "constants.h"
#include "point.h"
#include <vector>

class GeometryBox {
public:
    GeometryBox();
    ~GeometryBox();

    void add(VerticesObject object);
    int numVerts() {return n;};
    int numTris() {return n / 3;};
    int bufferSize() {return n * sizeof(Point);};
    void upload(Point* p);
    void boundingBox(Point* bounds);
private:
    Point* verts;
    bool packed;
    int n;
    std::vector<VerticesObject> objects;
};

GeometryBox::GeometryBox() {
    verts = nullptr;
    packed = false;
    n = 0;
    objects = std::vector<VerticesObject>();
}

GeometryBox::~GeometryBox() {
    if (packed) free(verts);
}

void GeometryBox::add(VerticesObject object) {
    objects.push_back(object);
    n += object.n;
    if (packed) {
        free(verts);
        packed = false;
    }
}

void GeometryBox::upload(Point* p) {
    if (!packed) {
        verts = (Point*)malloc(n * sizeof(Point));
        int i = 0;
        for (VerticesObject object : objects) {
            memcpy(&verts[i], object.verts, object.n * sizeof(Point));
            i += object.n;
        }
        packed = true;
    }
    HANDLE_CUDA_ERROR(cudaMemcpy(p, verts, n * sizeof(Point), cudaMemcpyHostToDevice));
}

void GeometryBox::boundingBox(Point* bounds) {
    for (VerticesObject object : objects) {
        for (int i = 0; i < object.n; i++) {
            Point v = object.verts[i];
            bounds[0] = min(v, bounds[0]);
            bounds[1] = max(v, bounds[1]);
        }
    }
}
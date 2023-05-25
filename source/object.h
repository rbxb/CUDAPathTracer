#pragma once

#include "point.h"
#include <vector>

struct ExpandedObject {
    Point* verts;
    Point* tex_coords;
    Point* normals;
    bool has_tex_coods;
    bool has_normals;
    int n;
};

class IndexedObject {
public:
    ExpandedObject expand();

    Point* verts;
    int* v_indices;

    Point* tex_coords;
    int* t_indices;

    Point* normals;
    int* n_indices;

    bool has_tex_coods;
    bool has_normals;

    int n;
};

ExpandedObject IndexedObject::expand() {
    ExpandedObject ex = ExpandedObject();
    ex.n = n;
    ex.has_tex_coods = has_tex_coods;
    ex.has_normals = has_normals;

    ex.verts = (Point*)malloc(sizeof(Point) * n);
    for (int i = 0; i < n; i++) ex.verts[i] = verts[v_indices[i]];

    if (has_tex_coods) {
        ex.tex_coords = (Point*)malloc(sizeof(Point) * n);
        for (int i = 0; i < n; i++) ex.tex_coords[i] = tex_coords[t_indices[i]];
    }

    if (has_normals) {
        ex.normals = (Point*)malloc(sizeof(Point) * n);
        for (int i = 0; i < n; i++) ex.normals[i] = normals[n_indices[i]];
    }

    return ex;
}
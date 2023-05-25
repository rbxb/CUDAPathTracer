#pragma once

#define MTL_HAS_TEXTURE 1

struct Material {
    float* texture;
    int tex_width;
    int tex_height;

    Point color;

    char flags;
}

__host__ __device__
Point get_mtl_color(const Material& mtl, float u, float v) {
    if (mtl.flags & MTL_HAS_TEXTURE) {
        int i = int((1 - v) * mtl.tex_height) * mtl.tex_width + int(u * mtl.tex_width);
        Point color;
        memcpy(&color, &mtl.texture[i*3], sizeof(Point));
        return color;
    } else {
        return mtl.color;
    }
}
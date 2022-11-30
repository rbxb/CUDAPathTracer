#pragma once

#include <cstdio>
#include <assert.h>
#include "point.h"

// Error handling function
#define HANDLE_CUDA_ERROR(error) if (error != cudaSuccess) {printf("\nCUDA error %d: %s in \"%s\" on line %d\n", error, cudaGetErrorName(error), __FILE__, __LINE__); exit(error);}

void init(bool print) {
    int device;
    HANDLE_CUDA_ERROR(cudaGetDevice(&device));

    struct cudaDeviceProp props;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&props, device));

    if (print) printf("%s Compute Capability %d.%d\n", props.name, props.major, props.minor);

    assert(sizeof(Point) == sizeof(float) * 3);
    assert(sizeof(Ray) == sizeof(float) * 6);
}
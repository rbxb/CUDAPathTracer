#pragma once

#include <cstdio>

// Error handling function
#define HANDLE_CUDA_ERROR(error) if (error != cudaSuccess) {printf("\nCUDA error %d: %s in \"%s\" on line %d\n", error, cudaGetErrorName(error), __FILE__, __LINE__); exit(error);}

void verifyCUDA(bool print) {
    int device;
    HANDLE_CUDA_ERROR(cudaGetDevice(&device));

    struct cudaDeviceProp props;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&props, device));

    if (print) printf("%s Compute Capability %d.%d\n", props.name, props.major, props.minor);
}
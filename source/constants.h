#pragma once

#include <cstdio>

// Error handling function
#define HANDLE_CUDA_ERROR(error) if (error != cudaSuccess) {printf("\nCUDA error %d: %s in \"%s\" on line %d\n", error, cudaGetErrorName(error), __FILE__, __LINE__); exit(error);}
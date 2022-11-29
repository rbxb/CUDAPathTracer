#include "constants.h"
#include "prefixsum.cu"

__global__
void prefixSumKernel(int* d_numbers, int n) {
    int i = threadIdx.x;
    sharedPrefixSum(i, d_numbers, n);
}

bool testPrefixSum() {
    const int n = 27;

    int* h_numbers = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_numbers[i] = i + 1;

    int* d_numbers;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_numbers, n * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_numbers, h_numbers, n * sizeof(int), cudaMemcpyHostToDevice));

    prefixSumKernel<<<1, n>>>(d_numbers, n);

    HANDLE_CUDA_ERROR(cudaMemcpy(h_numbers, d_numbers, n * sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = true;

    for (int i = 0; i < n; i++) {
        int k = i + 1;
        int expected = k * (k + 1) / 2;
        if (h_numbers[i] != expected) {
            passed = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR(cudaFree(d_numbers));
    free(h_numbers);

    return passed;
}
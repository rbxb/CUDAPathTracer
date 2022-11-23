#include <stdio.h>

#include "constants.h"
#include "prefixsum.cu"
#include "test_bucket.cu"

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

    for (int i = 0; i < n; i++) {
        int k = i + 1;
        int expected = k * (k + 1) / 2;
        if (h_numbers[i] != expected) return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    verifyCUDA(true);
    printf("\n");

    int passed = 0;
    int total = 0;

    printf("Test prefix sum: %s\n", testPrefixSum() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test bucket: %s\n", testBucket() ? "OK" + (passed++ * 0) : "FAILED"); total++;

    printf("\nPassed %d / %d\n", passed, total);
}
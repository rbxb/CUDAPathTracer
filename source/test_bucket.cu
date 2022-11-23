#include "constants.h"
#include "bucket.h"

#include <new>

#define TEST_BUCKET_PRINT false

__global__
void testBucketKernel(BucketDescriptor* d_desc, int* values, int numValues) {
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int bdim = blockDim.x;

    extern __shared__ char shared[];

    // Create the instance of BucketWriter in shared memory
    BucketWriter* bucket = (BucketWriter*)shared;
    if (tid == 0) {
        new (shared) BucketWriter(d_desc, block, bdim);
        bucket->setBuffer(&shared[sizeof(BucketWriter)]);
    }
    __syncthreads();
    
    // Iterate over input values and add them to buckets
    int numTiles = (numValues - 1) / bdim + 1;
    for (int tile = 0; tile < numTiles; tile++) {
        int x = tile * bdim + tid;
        int value = 999;
        if (x < numValues) value = values[x];
        __syncthreads();

        bucket->write(tid, x < numValues, value);
    }

    // One last flush for any remaining buffered data
    bucket->flush(tid);
}

bool testBucket() {
    const int CAP = 12;
    const int BLOCKSIZE = 8;
    const int MAX_BUCKETS = 16;

    int* d_data;
    int* d_headers;
    int* d_index;

    HANDLE_CUDA_ERROR(cudaMalloc(&d_data, CAP * MAX_BUCKETS * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_headers, MAX_BUCKETS * sizeof(int) * 2));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_index, sizeof(int)));

    int* d_values;

    const int NUM_VALUES = 100;

    HANDLE_CUDA_ERROR(cudaMalloc(&d_values, NUM_VALUES * sizeof(int)));

    int* h_values = (int*)malloc(NUM_VALUES * sizeof(int));

    for (int i = 0; i < NUM_VALUES; i++) {
        h_values[i] = i;
    }

    HANDLE_CUDA_ERROR(cudaMemcpy(d_values, h_values, NUM_VALUES * sizeof(int), cudaMemcpyHostToDevice));

    BucketDescriptor desc;
    desc.cap = CAP;
    desc.index = d_index;
    desc.headers = d_headers;
    desc.data = d_data;

    BucketDescriptor* d_desc;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_desc, sizeof(BucketDescriptor)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_desc, &desc, sizeof(BucketDescriptor), cudaMemcpyHostToDevice));

    testBucketKernel<<<1, BLOCKSIZE, BucketWriter::size(CAP, BLOCKSIZE)>>>(d_desc, d_values, NUM_VALUES);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    int numBuckets = 0;
    int* h_data = (int*)calloc(CAP * MAX_BUCKETS, sizeof(int));
    int* h_headers = (int*)calloc(MAX_BUCKETS, sizeof(int));

    HANDLE_CUDA_ERROR(cudaMemcpy(&numBuckets, d_index, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(h_data, d_data, CAP * numBuckets * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(h_headers, d_headers, numBuckets * sizeof(int) * 2, cudaMemcpyDeviceToHost));

    if (TEST_BUCKET_PRINT) {
        printf("\n");
        for (int i = 0; i < numBuckets; i++) {
            printf("Bucket %d by Block %d size %d \n", i, h_headers[i * 2], h_headers[i * 2 + 1]);
            for (int k = 0; k < h_headers[i * 2 + 1]; k++) {
                printf("%d ", h_data[i * CAP + k]);
            }
            printf("\n\n");
        }
    }

    if (numBuckets != (NUM_VALUES - 1) / CAP + 1) return false;

    int expected = 0;

    for (int i = 0; i < numBuckets; i++) {
        for (int k = 0; k < h_headers[i * 2 + 1]; k++) {
            if (h_data[i * CAP + k] != expected) return false;
            expected++;
        }
    }

    return true;
}
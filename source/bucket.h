#pragma once

#include "prefixsum.cu"
#include <cstdint>

typedef struct {
    int cap;
    int* index;

    int* headers;
    int* data;
} BucketDescriptor;

// BucketWriter helps create buckets
// A BucketWriter is instantiated in shared memory
//
// BucketWriter uses a circular buffer in shared memory to batch writes to global memory
// When the buffer is full (contains a bucket-worth of data), the buffer is flushed to global memory
// 
// Example of usage in test_bucket.cu
class BucketWriter {
public:
    // Constructor must only be called by one thread only
    // block is the index of the block
    // bdim is the blocksize
    __device__ BucketWriter(BucketDescriptor* desc, int block, int bdim);

    // Returns the size of the BucketDescriptor plus the size of the shared buffer
    // cap is the capacity of a bucket
    // bdim is the block size
    __device__ __host__ static int size(int cap, int bdim);

    // Sets the shared buffer
    // Must only be called by one thread
    __device__ void setBuffer(char* p);

    // Writes to the shared buffer and flushes it to global memory if it is full
    // Must be called by all threads in the block synchronously
    // write indicates whether or not this thread has a value that should be added to the bucket
    __device__ void write(int tid, bool write, int value);

    // Flushes any data in the shared buffer into global memory
    // Must be called by all threads in the block synchronously
    __device__ void flush(int tid);

private:
    BucketDescriptor desc;

    int block;
    int bdim;

    uint16_t* sums;
    int* buffer;
    int bufferCap;
    int len;
    int offset;

    int* globalData;
};

__device__
BucketWriter::BucketWriter(BucketDescriptor* desc, int block, int bdim) {
    memcpy(&this->desc, desc, sizeof(BucketDescriptor));
    this->block = block;
    this->bdim = bdim;
    bufferCap = max(bdim, this->desc.cap) + bdim;
    len = 0;
    offset = 0;
}

__device__ __host__
int BucketWriter::size(int cap, int bdim) {
    return sizeof(BucketWriter) + bdim * sizeof(uint16_t) + (cap + bdim) * sizeof(int);
}

__device__
void BucketWriter::setBuffer(char* p) {
    this->sums = (uint16_t*)p;
    this->buffer = (int*)&p[bdim * sizeof(uint16_t)];
}

__device__
void BucketWriter::write(int tid, bool write, int value) {
    // Get the index to write to using prefix sum
    sums[tid] = write? 1:0;
    sharedPrefixSum(tid, &sums[0], bdim);
    __syncthreads();

    // Write to shared buffer
    if (write) {
        int p = (offset + len + sums[tid] - 1) % bufferCap;
        buffer[p] = value;
    }
    __syncthreads();

    // Update fullness of shared buffer
    if (tid == 0) {
        len += sums[bdim - 1];
    }
    __syncthreads();

    // Flush to global memory if the shared buffer is full
    while (len >= desc.cap) {
        flush(tid);
    }
}

__device__
void BucketWriter::flush(int tid) {
    int flushLen = min(len, desc.cap);
    if (flushLen <= 0) return;   

    // Aquire new bucket
    if (tid == 0) {
        // Use atomic add to get the index of the next bucket
        int bucketIndex = atomicAdd(desc.index, 1);
        desc.headers[bucketIndex * 2] = block;
        desc.headers[bucketIndex * 2 + 1] = flushLen;
        globalData = &desc.data[bucketIndex * desc.cap];
    }
    __syncthreads();

    // Coalesced write to global memory
    for (int x = tid; x < flushLen; x += bdim) {
        int p = (offset + x) % bufferCap;
        globalData[x] = buffer[p];
    }
    __syncthreads();

    // Update offset and length of shared buffer
    if (tid == 0) {
        offset = (offset + flushLen) % bufferCap;
        len -= flushLen;
    }
    __syncthreads();
}
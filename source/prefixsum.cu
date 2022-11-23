#pragma once

template<typename T>
__device__
inline void sharedPrefixSum(int tid, T* list, int n) {
    for (int stride = 1; tid >= stride && stride < n; stride *= 2) {
        list[tid] += list[tid - stride];
    }
    __syncthreads();
}

template<typename T>
__device__
inline void efficientPrefixSum(int tid, T* list, int n) {
    int blocksize = n / 2;
    for (int stride = 1; stride <= blocksize; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * blocksize) list[index] += list[index - stride];
        __syncthreads();
    }

    for (int stride = blocksize / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < 2 * blocksize) list[index + stride] += list[index];
    }
}
#pragma once

template<typename T>
__device__
void sharedUnpack(const int& tid, const int& bdim, const int& n, T* global, T* shared) {
    const int SIZE_IN_FLOATS = sizeof(T) / sizeof(float);

    float* shared_f = (float*)shared;
    float* global_f = (float*)global;

    for (int x = tid; x < n * SIZE_IN_FLOATS; x += bdim) {
        shared_f[x] = global_f[x];
    }
}
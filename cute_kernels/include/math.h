#pragma once

#include <cuda.h>

template <typename T>
inline __host__ __device__ T ceil_divide(const T &x, const T &d) {
    return (x / d) + (x % d != 0);
}

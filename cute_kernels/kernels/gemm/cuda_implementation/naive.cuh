#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "index.h"

template <typename scalar_t>
inline __device__ void _run_matmul(const scalar_t *a,
                                   const scalar_t *b,
                                   const scalar_t *c,
                                   scalar_t *output,
                                   const bool &is_a_transposed,
                                   const bool &is_b_transposed,
                                   const fp32 &alpha,
                                   const fp32 &beta,
                                   const uint32 &i,
                                   const uint32 &j,
                                   const uint32 &M,
                                   const uint32 &K,
                                   const uint32 &N) {
    if (i < M && j < N) {
        fp32 accumulator = 0;
        for (uint32 k = 0; k < K; k++) {
            const uint64 a_index = get_matrix_index(i, k, M, K, is_a_transposed);
            const uint64 b_index = get_matrix_index(k, j, K, N, is_b_transposed);

            accumulator += a[a_index] * b[b_index];
        }

        accumulator *= alpha;
        const uint64 index = get_matrix_index(i, j, M, N, false);

        if (beta != 0) {
            accumulator += beta * c[index];
        }

        output[index] = accumulator;
    }
}

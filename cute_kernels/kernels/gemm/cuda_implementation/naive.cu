#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"
#include "index.h"

template <typename scalar_t>
__global__ void _naive_gemm_cuda_kernel(const scalar_t *a,
                                        const scalar_t *b,
                                        scalar_t *c,
                                        const bool is_a_transposed,
                                        const bool is_b_transposed,
                                        const uint32 M,
                                        const uint32 K,
                                        const uint32 N) {
    const uint32 i = blockDim.y * blockIdx.y + threadIdx.y;
    const uint32 j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < M && j < N) {
        fp32 accumulator = 0;
        for (uint32 k = 0; k < K; k++) {
            uint64 a_index;
            if (is_a_transposed) {
                a_index = get_matrix_index(k, i, M);
            } else {
                a_index = get_matrix_index(i, k, K);
            }

            uint64 b_index;
            if (is_b_transposed) {
                b_index = get_matrix_index(j, k, K);
            } else {
                b_index = get_matrix_index(k, j, N);
            }

            accumulator += a[a_index] * b[b_index];
        }

        c[get_matrix_index(i, j, N)] = accumulator;
    }
}

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const bool &is_a_transposed,
                     const bool &is_b_transposed,
                     const uint32 &M,
                     const uint32 &K,
                     const uint32 &N,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N) {
    TORCH_CHECK(BLOCK_SIZE_M % WARP_SIZE == 0);
    TORCH_CHECK(BLOCK_SIZE_N % WARP_SIZE == 0);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        a.scalar_type(), "naive_gemm_cuda_kernel", ([&] {
            dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(M, BLOCK_SIZE_M), ceil_divide<uint32>(N, BLOCK_SIZE_N), 1);
            dim3 BLOCK_SIZE = dim3(BLOCK_SIZE_M, BLOCK_SIZE_N, 1);

            _naive_gemm_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(a.data_ptr<scalar_t>(),
                                                                          b.data_ptr<scalar_t>(),
                                                                          c.data_ptr<scalar_t>(),
                                                                          is_a_transposed,
                                                                          is_b_transposed,
                                                                          M,
                                                                          K,
                                                                          N);
        }));
}

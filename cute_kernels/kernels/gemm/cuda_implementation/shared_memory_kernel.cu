#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"
#include "index.cuh"

namespace ck = cute_kernels;
namespace ck_mem = cute_kernels::memory;

using uint32 = ck::uint32;
using fp32 = ck::fp32;

template <typename scalar_t>
__global__ void _shared_memory_gemm_cuda_kernel(const scalar_t *A,
                                                const scalar_t *B,
                                                const scalar_t *C,
                                                scalar_t *output,
                                                const fp32 alpha,
                                                const fp32 beta,
                                                const uint32 M,
                                                const uint32 K,
                                                const uint32 N) {
    const uint32 i = blockIdx.y * blockDim.x + threadIdx.y;
    const uint32 j = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t *shared_memory = ck_mem::get_dynamic_shared_memory<scalar_t>();

    scalar_t *A_shared = shared_memory;
    scalar_t *B_shared = &shared_memory[blockDim.x * blockDim.x];

    fp32 accumulator = 0;

    // clang-format off
    #pragma unroll 128
    // clang-format on
    for (uint32 k = 0; k < K; k += blockDim.x) {
        const uint32 index = get_matrix_index<uint32, false>(threadIdx.y, threadIdx.x, blockDim.x, blockDim.x);

        // instead of looping over k dimension, we use the threads in the block to load the data to shared memory
        uint32 k_offset = k + threadIdx.x;
        if (i < M && k_offset < K) {
            A_shared[index] = A[get_matrix_index<uint32, false>(i, k_offset, M, K)];
        }

        // instead of looping over k dimension, we use the threads in the block to load the data to shared memory
        k_offset = k + threadIdx.y;
        if (j < N && k_offset < K) {
            B_shared[index] = B[get_matrix_index<uint32, false>(k_offset, j, K, N)];
        }

        __syncthreads();

        if (i < M && j < N) {
            const uint32 max_q = min(K - k, blockDim.x);
            for (uint32 q = 0; q < max_q; q++) {
                accumulator += A_shared[get_matrix_index<uint32, false>(threadIdx.y, q, blockDim.x, blockDim.x)] *
                               B_shared[get_matrix_index<uint32, false>(q, threadIdx.x, blockDim.x, blockDim.x)];
            }
        }

        // needed for ensuring that shared memory buffers are not modified before the loop finishes for all threads
        __syncthreads();
    }

    if (i < M && j < N) {
        accumulator *= alpha;
        const uint32 index = get_matrix_index<uint32, false>(i, j, M, N);

        if (beta != 0) {
            accumulator += beta * C[index];
        }

        output[index] = accumulator;
    }
}

void shared_memory_gemm_cuda(const torch::Tensor &A,
                             const torch::Tensor &B,
                             std::optional<torch::Tensor> &C,
                             torch::Tensor &output,
                             const bool &is_A_transposed,
                             const bool &is_B_transposed,
                             const fp32 &alpha,
                             const fp32 &beta,
                             const uint32 &M,
                             const uint32 &K,
                             const uint32 &N,
                             const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(A);
    CHECK_CUDA_TENSOR(B);
    CHECK_CUDA_TENSOR(output);

    if (C.has_value()) {
        CHECK_CUDA_TENSOR(C.value());
    }

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(!is_A_transposed);
    TORCH_CHECK(!is_B_transposed);

    dim3 NUM_BLOCKS = dim3(ck::ceil_divide<uint32>(N, BLOCK_SIZE), ck::ceil_divide<uint32>(M, BLOCK_SIZE), 1);
    dim3 BLOCK_SIZE_dim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

    DISPATCH_FLOAT_KERNEL(A.scalar_type(), "shared_memory_gemm_cuda_kernel", scalar_t, ([&] {
                              _shared_memory_gemm_cuda_kernel<scalar_t>
                                  <<<NUM_BLOCKS, BLOCK_SIZE_dim, 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(scalar_t)>>>(
                                      A.data_ptr<scalar_t>(),
                                      B.data_ptr<scalar_t>(),
                                      C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                                      output.data_ptr<scalar_t>(),
                                      alpha,
                                      beta,
                                      M,
                                      K,
                                      N);
                          }));
}

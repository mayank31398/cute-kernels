#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/launch.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"

#define MAX_ALLOWED_C 16384

namespace cg = cooperative_groups;

inline __device__ void _initialize_global_output(uint32 *output, const uint32 &C, const uint32 &global_thread_id) {
    const uint32 C4 = C >> 2;
    for (uint32 i = global_thread_id; i < C4; i += gridDim.x * blockDim.x) {
        ((uint32_4 *)output)[i] = DType<uint32>::make4(0, 0, 0, 0);
    }

    const uint32 index = (C4 << 2) + global_thread_id;
    if (index < C) {
        output[index] = 0;
    }
}

template <typename scalar_t>
inline __device__ void _update_local_count(const scalar_t *x,
                                           uint32 *shared_memory,
                                           const uint32 &num_elements,
                                           const uint32 &global_thread_id) {
    const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
    const uint32 num_elements4 = num_elements / num_elements_per_thread;

    for (uint32 i = global_thread_id; i < num_elements4; i += gridDim.x * blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, uint32> || std::is_same_v<scalar_t, int32>) {
            uint32_4 _x = ((uint32_4 *)x)[i];

            atomicAdd(&shared_memory[_x.x], 1);
            atomicAdd(&shared_memory[_x.y], 1);
            atomicAdd(&shared_memory[_x.z], 1);
            atomicAdd(&shared_memory[_x.w], 1);
        } else if constexpr (std::is_same_v<scalar_t, uint64> || std::is_same_v<scalar_t, int64>) {
            uint64_2 _x = ((uint64_2 *)x)[i];

            atomicAdd(&shared_memory[_x.x], 1);
            atomicAdd(&shared_memory[_x.y], 1);
        }
    }

    const uint32 index = (num_elements4 * num_elements_per_thread) + global_thread_id;
    if (index < num_elements) {
        atomicAdd(&shared_memory[x[index]], 1);
    }
}

template <typename scalar_t>
__global__ void _continuous_count_and_sort_cuda_kernel(const scalar_t *x,
                                                       uint32 *count_output,
                                                       uint32 *sorted_output,
                                                       uint32 *argsort_output,
                                                       const uint32 num_elements,
                                                       const uint32 C) {
    const uint32 local_thread_id = get_local_thread_id();
    const uint32 global_thread_id = get_global_thread_id();

    extern __shared__ uint32 shared_memory[];

    for (uint32 i = local_thread_id; i < C; i += blockDim.x) {
        shared_memory[i] = 0;
    }

    _initialize_global_output(count_output, C, global_thread_id);
    cg::this_grid().sync();

    _update_local_count<scalar_t>(x, shared_memory, num_elements, global_thread_id);

    __syncthreads();

    // write the output to the global memory and also record how many of the elements have been written before
    for (uint32 i = local_thread_id; i < C; i += blockDim.x) {
        shared_memory[i + C] = atomicAdd(&count_output[i], shared_memory[i]);
    }
}

void continuous_count_and_sort_cuda(const torch::Tensor &x,
                                    torch::Tensor &count_output,
                                    torch::Tensor &sorted_output,
                                    torch::Tensor &argsort_output,
                                    const uint32 &sm_count,
                                    const uint32 &C,
                                    const uint32 &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C <= MAX_ALLOWED_C);

    const uint64 num_elements = x.numel();
    assert(num_elements <= std::numeric_limits<uint32>::max() - 3);

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "continuous_count_and_sort_cuda_kernel", ([&] {
                                     const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                     auto [NUM_BLOCKS, _] = get_num_blocks<uint32>(
                                         num_elements, BLOCK_SIZE, num_elements_per_thread, sm_count);

                                     cudaFuncSetAttribute(_continuous_count_and_sort_cuda_kernel<scalar_t>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          2 * MAX_ALLOWED_C * sizeof(uint32));

                                     // dynamically sized clusters need this stupid way of launching the kernel
                                     cudaLaunchConfig_t launch_config = {0};
                                     launch_config.blockDim = BLOCK_SIZE;
                                     launch_config.gridDim = NUM_BLOCKS;
                                     // 2x shared memory since we need to sort as well
                                     launch_config.dynamicSmemBytes = 2 * C * sizeof(uint32);

                                     cudaLaunchAttribute attributes[1];
                                     attributes[0].id = cudaLaunchAttributeCooperative;
                                     attributes[0].val.cooperative = 1;

                                     launch_config.attrs = attributes;
                                     launch_config.numAttrs = 1;

                                     cudaLaunchKernelEx(&launch_config,
                                                        _continuous_count_and_sort_cuda_kernel<scalar_t>,
                                                        x.data_ptr<scalar_t>(),
                                                        count_output.data_ptr<uint32>(),
                                                        sorted_output.data_ptr<uint32>(),
                                                        argsort_output.data_ptr<uint32>(),
                                                        num_elements,
                                                        C);
                                 }));
}

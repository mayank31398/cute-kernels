#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/launch.h"
#include "../../../../include/math.h"
#include "../../../../include/threads.h"

template <typename scalar_t>
__global__ void _add_tensor_cuda_kernel(const scalar_t *x,
                                        const scalar_t *y,
                                        scalar_t *output,
                                        const uint64 num_elements) {
    constexpr int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 1 || vector_instruction_width == 2 || vector_instruction_width == 4 ||
                  vector_instruction_width == 8);

    using dtype = DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    const uint32 thread_id = get_global_thread_id();
    uint32 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (end < num_elements) {
        fp32_4 *output_vec = (fp32_4 *)output;

        if constexpr (std::is_same_v<scalar_t, fp32>) {
            const fp32 *x_vec = (fp32 *)&((fp32_4 *)x)[thread_id];
            const fp32 *y_vec = (fp32 *)&((fp32_4 *)y)[thread_id];
            fp32 output_buffer[vector_instruction_width];

            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = 0; i < vector_instruction_width; i++) {
                output_buffer[i] = x_vec[i] + y_vec[i];
            }

            output_vec[thread_id] = dtype::make4(output_buffer);
        } else {
            const fp32 *x_vec = (fp32 *)&((fp32_4 *)x)[thread_id];
            const fp32 *y_vec = (fp32 *)&((fp32_4 *)y)[thread_id];

            constexpr int n = vector_instruction_width >> 1;
            fp32 output_buffer[n];

            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = 0; i < n; i++) {
                T2 _x = dtype::reinterpret_32_bits_as_2x16(x_vec[i]);
                T2 _y = dtype::reinterpret_32_bits_as_2x16(y_vec[i]);

                _x = __hadd2(_x, _y);
                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(_x);
            }

            output_vec[thread_id] = DType<fp32>::make4(output_buffer);
        }
    }

    end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
    if (end < num_elements) {
        output[end] = x[end] + y[end];
    }
}

void add_tensor_cuda(const torch::Tensor &x, const torch::Tensor &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_tensor_cuda_kernel", ([&] {
            const uint32 vector_instruction_width = 16 / sizeof(scalar_t);

            std::vector<ChunkedArray<scalar_t>> x_chunks =
                chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> y_chunks =
                chunk_array<scalar_t>(y.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> output_chunks =
                chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

            for (int i = 0; i < x_chunks.size(); i++) {
                ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                ChunkedArray<scalar_t> y_chunk = y_chunks[i];
                ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                const uint32 num_elements = x_chunk.num_elements;

                const uint32 num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
                const uint32 NUM_BLOCKS = ceil_divide<uint64>(num_elements, num_elements_per_block);

                if constexpr (std::is_same_v<scalar_t, fp32>) {
                    _add_tensor_cuda_kernel<scalar_t>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y_chunk.array, output_chunk.array, num_elements);
                } else {
                    _add_tensor_cuda_kernel<scalar_t>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y_chunk.array, output_chunk.array, num_elements);
                }
            }
        }));
}

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/activations.h"
#include "../../../include/dtypes/all.h"
#include "../../../include/threads.h"

template <typename scalar_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const uint64 num_elements) {
    constexpr int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 4 || vector_instruction_width == 8);

    const uint64 thread_id = get_global_thread_id();
    using dtype = DType<scalar_t>;

    uint64 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (end < num_elements) {
        const fp32 *gate_vec = (fp32 *)&((fp32_4 *)gate)[thread_id];
        const fp32 *up_vec = (fp32 *)&((fp32_4 *)up)[thread_id];
        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = up_vec[i] * gate_vec[i] * sigmoid<fp32, fp32>(gate_vec[i]);
            } else {
                fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(gate_vec[i]));
                fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(up_vec[i]));

                _gate_upcast = DType<fp32>::make2(_up_upcast.x * _gate_upcast.x * sigmoid<fp32, fp32>(_gate_upcast.x),
                                                  _up_upcast.y * _gate_upcast.y * sigmoid<fp32, fp32>(_gate_upcast.y));

                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_upcast));
            }
        }

        ((fp32_4 *)output)[thread_id] = DType<fp32>::make4(output_buffer);
    }

    // use first warp for computing the last elements
    if (thread_id < WARP_SIZE) {
        // NOTE end is same as start since we don't use vector load stores here
        end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
        if (end < num_elements) {
            fp32 _gate_upcast = dtype::upcast(gate[end]);

            // up is upcasted automatically
            _gate_upcast = up[end] * _gate_upcast * sigmoid<fp32, fp32>(_gate_upcast);
            output[end] = dtype::downcast(_gate_upcast);
        }
    }
}

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor &output,
                         const int &BLOCK_SIZE) {
    const uint64 num_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_forward_cuda_kernel", ([&] {
            int log_vector_instruction_width;
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                log_vector_instruction_width = 2;
            } else {
                log_vector_instruction_width = 3;
            }

            const int num_elements_per_block = BLOCK_SIZE << log_vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));
}

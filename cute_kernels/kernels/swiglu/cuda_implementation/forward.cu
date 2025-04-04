#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename scalar_t>
inline __device__ scalar_t _swiglu_forward(const scalar_t &gate, const scalar_t &up) {
    using dtype = ck::DType<scalar_t>;

    fp32 _up = dtype::upcast(up);
    fp32 _gate = dtype::upcast(gate);
    fp32 _sigmoid = ck::sigmoid<fp32, fp32>(_gate);

    _sigmoid *= _gate * _up;

    return dtype::downcast(_sigmoid);
}

template <typename scalar_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const uint64 num_elements) {
    constexpr uint32 num_elements_per_thread = ck_mem::Packed128<scalar_t>::size;

    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 num_vector_elements = num_elements / num_elements_per_thread;

    if (thread_id < num_vector_elements) {
        // packed array allows loading using vector loads, its just a syntactic sugar
        const ck_mem::Packed128<const scalar_t> gate_vec = ck_mem::Packed128Array<const scalar_t>(gate)[thread_id];
        const ck_mem::Packed128<const scalar_t> up_vec = ck_mem::Packed128Array<const scalar_t>(up)[thread_id];
        ck_mem::Packed128<scalar_t> output_buffer;

        for (uint32 i = 0; i < num_elements_per_thread; i++) {
            output_buffer[i] = _swiglu_forward<scalar_t>(gate_vec[i], up_vec[i]);
        }

        ck_mem::Packed128Array<scalar_t> output_vec = ck_mem::Packed128Array<scalar_t>(output);
        output_vec[thread_id] = output_buffer;
    }

    const uint32 index = num_vector_elements * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        output[index] = _swiglu_forward<scalar_t>(gate[index], up[index]);
    }
}

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor &output,
                         const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(gate);
    CHECK_CUDA_TENSOR(up);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(gate.scalar_type(), "swiglu_forward_cuda_kernel", ([&] {
                                       const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                       const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

                                       std::vector<ck::ChunkedArray<scalar_t>> gate_chunks =
                                           ck::chunk_array<scalar_t>(gate.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> up_chunks =
                                           ck::chunk_array<scalar_t>(up.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                           ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                                       for (int i = 0; i < gate_chunks.size(); i++) {
                                           ck::ChunkedArray<scalar_t> gate_chunk = gate_chunks[i];
                                           ck::ChunkedArray<scalar_t> up_chunk = up_chunks[i];
                                           ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                           const uint64 num_elements = gate_chunk.num_elements;
                                           const uint32 NUM_BLOCKS =
                                               ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                                           _swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                               gate_chunk.array, up_chunk.array, output_chunk.array, num_elements);
                                       }
                                   }));
}

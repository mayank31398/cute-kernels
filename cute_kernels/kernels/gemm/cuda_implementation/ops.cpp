#include <torch/extension.h>

#include "../../../include/dtypes/alias.h"

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     torch::Tensor &output,
                     const bool &is_a_transposed,
                     const bool &is_b_transposed,
                     const float alpha,
                     const float beta,
                     const uint &M,
                     const uint &K,
                     const uint &N,
                     const uint &BLOCK_SIZE_M,
                     const uint &BLOCK_SIZE_N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("naive_gemm_cuda", &naive_gemm_cuda, "naive GEMM (CUDA)"); }

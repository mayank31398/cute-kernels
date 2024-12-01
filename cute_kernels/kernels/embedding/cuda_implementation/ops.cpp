#include <torch/extension.h>

void embedding_forward_cuda(const torch::Tensor &input_ids,
                            const torch::Tensor &weight,
                            torch::Tensor output,
                            const int &BLOCK_SIZE_B,
                            const int &BLOCK_SIZE_H);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward_cuda", &embedding_forward_cuda, "Embedding forward (CUDA)");
}

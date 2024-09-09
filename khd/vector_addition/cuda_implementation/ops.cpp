#include <torch/extension.h>

void vector_addition_forward_cuda_kernel(
    torch::Tensor x, torch::Tensor y, torch::Tensor output, const int num_elements, const int BLOCK_SIZE);

torch::Tensor vector_addition_forward_cuda(torch::Tensor x,
                                           torch::Tensor y,
                                           const bool in_place,
                                           const int BLOCK_SIZE) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU");
    TORCH_CHECK(y.device().is_cuda(), "tensor y is not on GPU");

    TORCH_CHECK(x.sizes() == y.sizes(), "tensors x and y should have same shape");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "tensors x and y should have same dtype");

    torch::Tensor output = x;
    if (!in_place) {
        output = torch::empty_like(x);
    }

    int num_elements = x.numel();

    vector_addition_forward_cuda_kernel(x.view(-1), y.view(-1), output.view(-1), num_elements, BLOCK_SIZE);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_addition_forward_cuda", &vector_addition_forward_cuda, "Vector addition forward (CUDA)");
}

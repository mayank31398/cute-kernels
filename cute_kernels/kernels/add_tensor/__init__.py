import torch

from ...math import ceil_divide
from ...utils import ensure_same_strides, is_nvidia_gpu
from .cuda_implementation import add_tensor_cuda
from .torch_implementation import add_tensor_torch
from .triton_implementation import _add_tensor_triton_kernel


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, kernel_backend: str) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y)
        output = torch.empty_like(x)
        BLOCK_SIZE = 1024

        if is_nvidia_gpu() and x.is_cuda and y.is_cuda:
            add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
        else:
            num_elements = x.numel()
            num_programs = ceil_divide(num_elements, BLOCK_SIZE=BLOCK_SIZE)

            with torch.cuda.device(x.device):
                _add_tensor_triton_kernel[(num_programs,)](
                    x_ptr=x, y_ptr=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
                )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def add_tensor_cute(x: torch.Tensor, y: torch.Tensor, kernel_backend: str | None = None) -> torch.Tensor:
    return _AddTensor_Cute.apply(x, y, kernel_backend)

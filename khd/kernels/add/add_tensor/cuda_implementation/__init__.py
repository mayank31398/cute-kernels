import torch

from .....kernel_registry import KernelRegistry


_KERNEL_NAME = "add_tensor_forward_cuda"

KernelRegistry.get_kernel(_KERNEL_NAME)


class _AddTensor_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert y.is_cuda, "tensor y is not on GPU"

        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        output = torch.empty_like(x)

        torch.ops.khd.add_tensor_forward_cuda(
            x=x, y=y, output=output, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None, None


def add_tensor_cuda(x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        vectorized_loop_size (int): vector instructions' operand size
        BLOCK_SIZE (int): block size

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_CUDA.apply(x, y, vectorized_loop_size, BLOCK_SIZE)

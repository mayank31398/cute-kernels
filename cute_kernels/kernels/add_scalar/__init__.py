import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...utils import is_nvidia_gpu
from .cuda_implementation import add_scalar_cuda
from .torch_implementation import add_scalar_torch
from .triton_implementation import add_scalar_triton


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: float,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_CUDA: int,
        BLOCK_SIZE_TRITON: int,
        NUM_WARPS_TRITON: int,
    ) -> torch.Tensor:
        output = torch.empty_like(x)

        if is_cuda_kernel_backend_allowed(kernel_backend) and is_nvidia_gpu() and x.is_cuda:
            add_scalar_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE_CUDA)
        elif is_triton_kernel_backend_allowed(kernel_backend):
            add_scalar_triton(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE_TRITON, NUM_WARPS=NUM_WARPS_TRITON)
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, *[None] * 5


def add_scalar_cute(
    x: torch.Tensor,
    y: float,
    *,
    kernel_backend: KernelBackend = KernelBackend.cuda,
    # cuda
    BLOCK_SIZE_CUDA: int = 1024,
    # triton
    BLOCK_SIZE_TRITON: int = 4096,
    NUM_WARPS_TRITON: int = 32,
) -> torch.Tensor:
    """adds a float value to a tensor

    Args:
        x (torch.Tensor): input tensor
        y (float): float value to add to `x`
        kernel_backend (KernelBackend, optional): kernel backend to prioritize. Defaults to KernelBackend.cuda.
        BLOCK_SIZE_CUDA (int, optional): block size for CUDA backend. Defaults to 1024.
        BLOCK_SIZE_TRITON (int, optional): block size for triton backend. Defaults to 4096.
        NUM_WARPS_TRITON (int, optional): warps for triton backend. Defaults to 32.

    Returns:
        torch.Tensor: output tensor
    """

    if y == 0:
        return x

    return _AddScalar_Cute.apply(x, y, kernel_backend, BLOCK_SIZE_CUDA, BLOCK_SIZE_TRITON, NUM_WARPS_TRITON)

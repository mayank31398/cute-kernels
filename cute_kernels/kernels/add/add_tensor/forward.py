import torch

from ....cutotune import cutotune
from ....enums import KernelBackend
from ..parameters import get_cutotune_parameters
from .cuda_implementation import add_tensor_cuda
from .triton_implementation import add_tensor_triton


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    output = torch.empty_like(x)

    if kernel_backend == KernelBackend.cuda:
        assert x.is_cuda, "tensor x is not on GPU"
        assert y.is_cuda, "tensor y is not on GPU"

        add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
    elif kernel_backend == KernelBackend.triton:
        add_tensor_triton(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output

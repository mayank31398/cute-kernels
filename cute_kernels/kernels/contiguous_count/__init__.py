import torch

from ...enums import KernelBackend
from ...utils import ensure_contiguous, get_sm_count
from .cuda_implementation import contiguous_count_cuda
from .torch_implementation import contiguous_count_torch
from .triton_implementation import contiguous_count_triton


@torch.no_grad()
@ensure_contiguous
def contiguous_count_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend = KernelBackend.triton,
    BLOCK_SIZE_B: int = 64,
) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    output = torch.zeros(size, dtype=torch.uint32, device=x.device)

    if kernel_backend == KernelBackend.cuda:
        contiguous_count_cuda(
            x=x, output=output, sm_count=get_sm_count(x.device), size=size, BLOCK_SIZE_B=BLOCK_SIZE_B
        )
    elif kernel_backend == KernelBackend.triton:
        contiguous_count_triton(x=x, output=output, size=size, BLOCK_SIZE_B=BLOCK_SIZE_B)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output

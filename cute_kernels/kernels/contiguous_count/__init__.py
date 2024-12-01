import torch
import triton

from ...enums import KernelBackend
from ...utils import ceil_divide, ensure_contiguous, get_sm_count
from .triton_implementation import contiguous_count_triton_kernel


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

    B = x.numel()
    C = size
    BLOCK_SIZE_C = triton.next_power_of_2(C)

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    output = torch.zeros(num_programs, C, dtype=torch.long, device=x.device)

    if kernel_backend == KernelBackend.triton:
        with torch.device(x.device):
            contiguous_count_triton_kernel[(num_programs,)](
                x_ptr=x,
                output_ptr=output,
                B=B,
                C=C,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_C=BLOCK_SIZE_C,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output.sum(dim=0)

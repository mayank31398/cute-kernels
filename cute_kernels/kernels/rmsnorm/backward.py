import torch
import triton

from ...constants import MAX_TRITON_BLOCK_SIZE, TORCH_TO_TRITON_DTYPE, TRITON_BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, CutoTuneParameter, ceil_divide, cutotune, get_sm_count
from .triton_implementation import rmsnorm_backward_triton_kernel


@cutotune(
    configs=[
        CutoTuneConfig(
            config={"BLOCK_SIZE_B": BLOCK_SIZE_B},
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"]
            <= MAX_TRITON_BLOCK_SIZE,
        )
        for BLOCK_SIZE_B in [1, 2, 4, 8, 16, 32] + TRITON_BLOCK_SIZES_POWERS_OF_2
    ],
    triggers={"x.dtype", "BLOCK_SIZE_H"},
)
def _triton_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    rmsnorm_denominator: torch.Tensor,
    x_grad: torch.Tensor,
    eps: float,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor | None:
    num_elements, hidden_size = x.size()

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(num_elements, BLOCK_SIZE_B))

    has_weight = weight is not None
    weight_grad = (
        torch.empty(num_programs, hidden_size, device=x_grad.device, dtype=torch.float32) if has_weight else None
    )

    with torch.device(x.device):
        rmsnorm_backward_triton_kernel[(num_programs,)](
            x_ptr=x,
            x_stride_b=x.stride(0),
            x_stride_h=x.stride(1),
            x_dtype=TORCH_TO_TRITON_DTYPE[x.dtype],
            has_weight=weight is not None,
            weight_ptr=weight,
            output_grad_ptr=output_grad,
            output_grad_stride_b=output_grad.stride(0),
            output_grad_stride_h=output_grad.stride(1),
            x_grad_ptr=x_grad,
            weight_grad_ptr=weight_grad,
            weight_grad_stride_b=weight_grad.stride(0) if has_weight else None,
            weight_grad_stride_h=weight_grad.stride(1) if has_weight else None,
            eps=eps,
            memory_efficient=rmsnorm_denominator is None,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

    if has_weight:
        weight_grad = weight_grad.sum(dim=0).type_as(weight)

    return weight_grad


@cutotune(configs=[CutoTuneConfig(config={"kernel_backend": KernelBackend.triton})], triggers={"x.dtype"})
def _backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rmsnorm_denominator: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int | CutoTuneParameter,
    BLOCK_SIZE_H: int | CutoTuneParameter,
) -> tuple[torch.Tensor | None]:
    hidden_size = x.size(-1)
    x_grad = torch.empty_like(x)

    if kernel_backend == KernelBackend.triton:
        # NOTE we ignore the BLOCK_SIZE_H passed by user
        BLOCK_SIZE_H = triton.next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        weight_grad = _triton_backward(
            x=x,
            weight=weight,
            output_grad=output_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            eps=eps,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad, weight_grad

import torch
import triton

from ...constants import MAX_TRITON_BLOCK_SIZE, TORCH_TO_TRITON_DTYPE, TRITON_BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, CutoTuneParameter, cutotune, ensure_same_strides
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
    triggers={"x_view.dtype", "BLOCK_SIZE_H"},
)
def _triton_backward(
    x_view: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    rmsnorm_denominator: torch.Tensor,
    x_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    eps: float,
    memory_efficient: bool,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements, hidden_size = x_view.size()

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    grid = (1,)

    with torch.device(x_view.device):
        rmsnorm_backward_triton_kernel[grid](
            x_ptr=x_view,
            x_stride_b=x_view.stride(0),
            x_stride_h=x_view.stride(1),
            x_dtype=TORCH_TO_TRITON_DTYPE[x_view.dtype],
            has_weight=weight is not None,
            weight_ptr=weight,
            output_grad_ptr=output_grad,
            output_grad_stride_b=output_grad.stride(0),
            output_grad_stride_h=output_grad.stride(1),
            x_grad_ptr=x_grad,
            weight_grad_ptr=weight_grad,
            eps=eps,
            memory_efficient=memory_efficient,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )


@cutotune(configs=[CutoTuneConfig(config={"kernel_backend": KernelBackend.triton})], triggers={"x.dtype"})
def _backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rmsnorm_denominator: torch.Tensor,
    output_grad: torch.Tensor,
    memory_efficient: bool,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int | CutoTuneParameter,
    BLOCK_SIZE_H: int | CutoTuneParameter,
) -> tuple[torch.Tensor | None]:
    # x already has stride(-1) = 1 from the forward function
    # so we just ensure that x & output_grad have the same strides
    x, output_grad = ensure_same_strides(x, output_grad)

    has_weight = weight is not None
    hidden_size = x.size(-1)

    x_grad = torch.empty_like(x)
    weight_grad = torch.empty(hidden_size, device=x.device, dtype=x.dtype) if has_weight else None

    x_view = x.view(-1, hidden_size)
    output_grad_view = output_grad.view(-1, hidden_size)

    if kernel_backend == KernelBackend.triton:
        BLOCK_SIZE_H = triton.next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        _triton_backward(
            x_view=x_view,
            weight=weight,
            output_grad=output_grad_view,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            weight_grad=weight_grad,
            eps=eps,
            memory_efficient=memory_efficient,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad, weight_grad

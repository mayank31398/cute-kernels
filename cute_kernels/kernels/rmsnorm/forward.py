import torch
import triton

from ...constants import MAX_TRITON_BLOCK_SIZE, TORCH_TO_TRITON_DTYPE, TRITON_BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, CutoTuneParameter, ceil_divide, cutotune
from .triton_implementation import rmsnorm_forward_triton_kernel


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
def _triton_forward(
    x_view: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    rmsnorm_denominator: torch.Tensor,
    eps: float,
    memory_efficient: bool,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements, hidden_size = x_view.size()

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    with torch.device(x_view.device):
        rmsnorm_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x_view,
            x_stride_b=x_view.stride(0),
            x_stride_h=x_view.stride(1),
            x_dtype=TORCH_TO_TRITON_DTYPE[x_view.dtype],
            has_weight=weight is not None,
            weight_ptr=weight,
            output_ptr=output,
            output_stride_b=output.stride(0),
            output_stride_h=output.stride(1),
            eps=eps,
            memory_efficient=memory_efficient,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )


@cutotune(configs=[CutoTuneConfig(config={"kernel_backend": KernelBackend.triton})], triggers={"x.dtype"})
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    memory_efficient: bool,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int | CutoTuneParameter,
    BLOCK_SIZE_H: int | CutoTuneParameter,
) -> tuple[torch.Tensor | None]:
    if x.stride(-1) != 1:
        x = x.contiguous()

    assert x.dim() > 1, "x should have more than 1 dimensions"

    has_weight = weight is not None

    if has_weight:
        assert weight.dim() == 1, "weight should be 1D"
        assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
        assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        weight = weight.contiguous()

    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size

    x_view = x.view(-1, hidden_size)

    output = torch.empty_like(x)
    rmsnorm_denominator = None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)

    if kernel_backend == KernelBackend.triton:
        BLOCK_SIZE_H = triton.next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        _triton_forward(
            x_view=x_view,
            weight=weight,
            output=output,
            rmsnorm_denominator=rmsnorm_denominator,
            eps=eps,
            memory_efficient=memory_efficient,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output, rmsnorm_denominator

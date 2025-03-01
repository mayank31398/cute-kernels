import torch

from ...cutotune import cutotune
from ...math import ceil_divide
from .cuda_implementation import swiglu_backward_cuda
from .parameters import get_cutotune_parameters
from .triton_implementation import _swiglu_backward_triton_kernel


@cutotune(**get_cutotune_parameters())
def _backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor]:
    gate_grad = torch.empty_like(gate)
    up_grad = torch.empty_like(up)

    if kernel_backend == "cuda":
        swiglu_backward_cuda(
            gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=BLOCK_SIZE
        )
    elif kernel_backend == "triton":
        num_elements = gate.numel()

        _swiglu_backward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_grad_ptr=output_grad,
            gate_grad_ptr=gate_grad,
            up_grad_ptr=up_grad,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return gate_grad, up_grad

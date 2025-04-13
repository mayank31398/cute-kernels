import torch

from ...cutotune import cutotune
from .parameters import get_cutotune_parameters
from .triton_implementation import _swiglu_unchunked_backward_triton_kernel


@cutotune(**get_cutotune_parameters())
def _backward(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor]:
    x_grad = torch.empty_like(x)

    if kernel_backend == "triton":
        B, H = get_num_elements_and_hidden_size(x)

        with torch.device(x.device):
            _swiglu_unchunked_backward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
                x_ptr=x,
                output_grad_ptr=output_grad,
                x_grad_ptr=x_grad,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad

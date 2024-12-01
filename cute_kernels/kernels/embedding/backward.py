import torch

from ...enums import KernelBackend
from ...utils import (
    CutoTuneConfig,
    ceil_divide,
    cutotune,
    get_block_sizes_powers_of_2,
    get_cartesian_product_cutotune_configs,
)
from .triton_implementation import embedding_backward_triton_kernel


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        BLOCK_SIZE_B=get_block_sizes_powers_of_2(128, 65536),
        BLOCK_SIZE_H=get_block_sizes_powers_of_2(128, 65536),
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= 65536,
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE_B": 128, "BLOCK_SIZE_H": 128}),
    triggers={"weight.dtype"},
)
def _backward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    hidden_size = weight.size(-1)
    num_elements = input_ids.numel()

    weight_grad = torch.zeros_like(weight)

    if kernel_backend == KernelBackend.triton:
        if weight.dtype == torch.bfloat16:
            raise NotImplementedError("bf16 is not supported with triton backend for backward for embeddings")

        with torch.device(input_ids.device):
            embedding_backward_triton_kernel[
                (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
            ](
                input_ids_ptr=input_ids,
                output_grad_ptr=output_grad,
                weight_grad_ptr=weight_grad,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return weight_grad

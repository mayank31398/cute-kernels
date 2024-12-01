import torch

from ...enums import KernelBackend
from ...utils import (
    CutoTuneConfig,
    ceil_divide,
    cutotune,
    get_block_sizes_powers_of_2,
    get_cartesian_product_cutotune_configs,
)
from .cuda_implementation import embedding_forward_cuda_kernel
from .triton_implementation import embedding_forward_triton_kernel


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.cuda],
        BLOCK_SIZE_B=get_block_sizes_powers_of_2(1, 32),
        BLOCK_SIZE_H=get_block_sizes_powers_of_2(32, 1024),
        condition=lambda **kwargs: kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= 1024,
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        BLOCK_SIZE_B=get_block_sizes_powers_of_2(128, 65536),
        BLOCK_SIZE_H=get_block_sizes_powers_of_2(128, 65536),
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= 65536,
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE_B": 128, "BLOCK_SIZE_H": 128}),
    triggers={"weight.dtype"},
)
def _forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    num_elements = input_ids.numel()
    hidden_size = weight.size(-1)

    output = torch.empty(num_elements, hidden_size, dtype=weight.dtype, device=input_ids.device)

    if kernel_backend == KernelBackend.cuda:
        assert input_ids.is_cuda
        assert weight.is_cuda

        embedding_forward_cuda_kernel(
            input_ids=input_ids,
            weight=weight,
            output=output,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    elif kernel_backend == KernelBackend.triton:
        with torch.device(input_ids.device):
            embedding_forward_triton_kernel[
                (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
            ](
                input_ids_ptr=input_ids,
                weight_ptr=weight,
                output_ptr=output,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output.view(*input_ids.size(), hidden_size)

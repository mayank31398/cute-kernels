import torch

from ....constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2, LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op, get_sm_count
from .kernels_forward import _contiguous_count_triton_kernel


_KERNEL_NAME = "contiguous_count_triton"


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        BLOCK_SIZE=get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE"] * kwargs["BLOCK_SIZE_C"] <= MAX_TRITON_BLOCK_SIZE,
    ),
    default_config=CutoTuneConfig({"BLOCK_SIZE": 1}),
    triggers={"x.dtype", "BLOCK_SIZE_C"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def contiguous_count_triton(
    x: torch.Tensor, output: torch.Tensor, size: int, BLOCK_SIZE: int, BLOCK_SIZE_C: int
) -> None:
    B = x.numel()

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE))

    with torch.device(x.device):
        _contiguous_count_triton_kernel[(num_programs,)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            C=size,
            BLOCK_SIZE_B=BLOCK_SIZE,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

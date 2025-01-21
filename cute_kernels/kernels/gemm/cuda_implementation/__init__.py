import torch

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....jit import cpp_jit
from ....math import get_powers_of_2
from ....utils import cute_op


_KERNEL_NAME = "naive_gemm_cuda"


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_M=get_powers_of_2(4, 32),
        BLOCK_SIZE_N=get_powers_of_2(4, 32),
        condition=lambda **kwargs: kwargs["BLOCK_SIZE_M"] * kwargs["BLOCK_SIZE_N"] >= 32,
    ),
    default_config=CutoTuneConfig(dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16)),
    triggers={"a.dtype", "is_a_transposed", "is_b_transposed"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"c"})
@cpp_jit(_KERNEL_NAME)
def naive_gemm_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    is_a_transposed: bool,
    is_b_transposed: bool,
    M: int,
    K: int,
    N: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> None: ...

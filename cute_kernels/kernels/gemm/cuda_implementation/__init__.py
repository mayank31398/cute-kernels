import torch

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....jit import cpp_jit
from ....math import get_powers_of_2
from ....utils import cute_op


_NAIVE_KERNEL_NAME = "naive_gemm_cuda"


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_M=get_powers_of_2(4, 32),
        BLOCK_SIZE_N=get_powers_of_2(4, 32),
        condition=lambda **kwargs: kwargs["BLOCK_SIZE_M"] * kwargs["BLOCK_SIZE_N"] >= 32,
    ),
    default_config=CutoTuneConfig(dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16)),
    triggers={"a.dtype", "is_a_transposed", "is_b_transposed"},
)
@cute_op(f"{LIBRARY_NAME}::{_NAIVE_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_NAIVE_KERNEL_NAME)
def naive_gemm_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None,
    output: torch.Tensor,
    is_a_transposed: bool,
    is_b_transposed: bool,
    alpha: float,
    beta: float,
    M: int,
    K: int,
    N: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> None: ...


@cutotune(
    get_cartesian_product_cutotune_configs(BLOCK_SIZE=get_powers_of_2(8, 32)),
    default_config=CutoTuneConfig(dict(BLOCK_SIZE=32)),
    triggers={"a.dtype", "is_a_transposed", "is_b_transposed"},
)
@cute_op(f"{LIBRARY_NAME}::{_NAIVE_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_NAIVE_KERNEL_NAME)
def shared_memory_gemm_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None,
    output: torch.Tensor,
    is_a_transposed: bool,
    is_b_transposed: bool,
    alpha: float,
    beta: float,
    M: int,
    K: int,
    N: int,
    BLOCK_SIZE: int,
) -> None: ...

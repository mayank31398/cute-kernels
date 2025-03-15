from typing import Callable

import tilelang.language as T
import torch

from ....constants import LIBRARY_NAME
from ....utils import cute_op


_KERNEL_NAME = "add_scalar_tilelang"


def _add_scalar_tilelang_kernel(
    num_elements: int, y: float, BLOCK_SIZE: int, dtype: torch.dtype, num_threads: int
) -> Callable:
    @T.prim_func
    def main(x: T.Buffer((num_elements,), dtype), output: T.Buffer((num_elements,), dtype)):
        with T.Kernel(T.ceildiv(num_elements, BLOCK_SIZE), 1, 1, threads=num_threads) as (bx, by, bz):
            for i in T.Parallel(BLOCK_SIZE):
                if bx * BLOCK_SIZE + i < num_elements:
                    output[bx * BLOCK_SIZE + i] = x[bx * BLOCK_SIZE + i] + y

    return main


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def add_scalar_tilelang(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int, num_threads: int) -> None:
    num_elements = x.numel()
    program = _add_scalar_tilelang_kernel(num_elements, 0.1, 1024, "float32", num_threads)
    program = tilelang.compile(program)

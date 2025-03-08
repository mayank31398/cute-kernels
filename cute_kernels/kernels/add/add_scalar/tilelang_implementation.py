from typing import Callable

import tilelang.language as T
import torch

from ....constants import LIBRARY_NAME
from ....utils import cute_op


_KERNEL_NAME = "add_scalar_tilelang"


def _add_scalar_tilelang_kernel(num_elements: int, y: float, BLOCK_SIZE: int, dtype: torch.dtype) -> Callable:
    @T.prim_func
    def main(x: T.Buffer((num_elements,), dtype), output: T.Buffer((num_elements,), dtype)):
        with T.Kernel(T.ceildiv(num_elements, BLOCK_SIZE)) as bx:
            T.alloc_fragment()
            T.copy()
            T.write()


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def add_scalar_tilelang(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    pass

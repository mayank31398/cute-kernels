from typing import Callable

import tilelang
import tilelang.language as T
import torch


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


num_elements = 1024
y = 0.1
BLOCK_SIZE = 1024
dtype = "float32"
num_threads = 1024

program = _add_scalar_tilelang_kernel(num_elements, 0.1, 1024, "float32", num_threads)
_add_scalar_tilelang_kernel_compiled = tilelang.compile(program)
print(_add_scalar_tilelang_kernel_compiled.get_kernel_source())

x = torch.randn(1024)
output = torch.empty_like(x)

_add_scalar_tilelang_kernel_compiled(x, output)

print(output - x)

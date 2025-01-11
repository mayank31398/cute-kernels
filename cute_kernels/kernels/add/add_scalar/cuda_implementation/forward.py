import torch

from .....constants import LIBRARY_NAME
from .....jit import cpp_jit
from .....utils import cute_op


_KERNEL_NAME = "add_scalar_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def add_scalar_cuda(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None: ...

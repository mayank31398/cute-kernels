import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import cuda_jit
from ....utils import cute_op


_KERNEL_NAME = "swiglu_forward_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cuda_jit
def swiglu_forward_cuda(
    gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None: ...

import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry
from ....utils import cute_op


_FORWARD_KERNEL_NAME = "embedding_forward_cuda"
_BACKWARD_KERNEL_NAME = "embedding_backward_cuda"


@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def embedding_forward_cuda_kernel(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(input_ids, weight, output, BLOCK_SIZE_B, BLOCK_SIZE_H)


@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"weight_grad"})
def embedding_backward_cuda_kernel(
    input_ids: torch.Tensor,
    output_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(input_ids, output_grad, weight_grad, BLOCK_SIZE_B, BLOCK_SIZE_H)

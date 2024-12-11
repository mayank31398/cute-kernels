import torch

from ...enums import KernelBackend
from ...utils import CutoTuneParameter, ensure_contiguous
from .backward import _backward
from .forward import _forward
from .torch_implementation import swiglu_unchunked_torch


class _SwigluUnchunked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
        vector_instruction_width_forward: int,
        vector_instruction_width_backward: int,
        BLOCK_SIZE_forward: int,
        BLOCK_SIZE_backward: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.vector_instruction_width_backward = vector_instruction_width_backward
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        return _forward(
            x=x,
            kernel_backend=kernel_backend_forward,
            vector_instruction_width=vector_instruction_width_forward,
            BLOCK_SIZE=BLOCK_SIZE_forward,
        )

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x = ctx.saved_tensors[0]

        x_grad = _backward(
            x=x,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
            vector_instruction_width=ctx.vector_instruction_width_backward,
            BLOCK_SIZE=ctx.BLOCK_SIZE_backward,
        )

        return x_grad, *[None] * 6


def swiglu_unchunked_cute(
    x: torch.Tensor,
    kernel_backend_forward: KernelBackend = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend = CutoTuneParameter(),
    vector_instruction_width_forward: int = CutoTuneParameter(),
    vector_instruction_width_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _SwigluUnchunked_Cute.apply(
        x,
        kernel_backend_forward,
        kernel_backend_backward,
        vector_instruction_width_forward,
        vector_instruction_width_backward,
        BLOCK_SIZE_forward,
        BLOCK_SIZE_backward,
    )

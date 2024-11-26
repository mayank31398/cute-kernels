import torch

from ...enums import KernelBackend
from ...utils import CutoTuneParameter, ensure_same_strides
from .backward import _backward
from .forward import _forward
from .torch_implementation import swiglu_torch


class _Swiglu_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend_forward: KernelBackend | CutoTuneParameter,
        kernel_backend_backward: KernelBackend | CutoTuneParameter,
        vector_instruction_width_forward: int | CutoTuneParameter,
        vector_instruction_width_backward: int | CutoTuneParameter,
        BLOCK_SIZE_forward: int | CutoTuneParameter,
        BLOCK_SIZE_backward: int | CutoTuneParameter,
    ) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        gate, up = ensure_same_strides(gate, up)

        ctx.save_for_backward(gate, up)

        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.vector_instruction_width_backward = vector_instruction_width_backward
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        return _forward(
            gate=gate,
            up=up,
            kernel_backend=kernel_backend_forward,
            vector_instruction_width=vector_instruction_width_forward,
            BLOCK_SIZE=BLOCK_SIZE_forward,
        )

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        gate, up, output_grad = ensure_same_strides(gate, up, output_grad)

        gate_grad, up_grad = _backward(
            gate=gate,
            up=up,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
            vector_instruction_width=ctx.vector_instruction_width_backward,
            BLOCK_SIZE=ctx.BLOCK_SIZE_backward,
        )

        return gate_grad, up_grad, *[None] * 6


def swiglu_cute(
    gate: torch.Tensor,
    up: torch.Tensor,
    kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    vector_instruction_width_forward: int | CutoTuneParameter = CutoTuneParameter(),
    vector_instruction_width_backward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_forward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_backward: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _Swiglu_Cute.apply(
        gate,
        up,
        kernel_backend_forward,
        kernel_backend_backward,
        vector_instruction_width_forward,
        vector_instruction_width_backward,
        BLOCK_SIZE_forward,
        BLOCK_SIZE_backward,
    )

import torch

from ...enums import KernelBackend
from ...utils import CutoTuneParameter
from .backward import _backward
from .forward import _forward
from .torch_implementation import rmsnorm_torch


class _RMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        memory_efficient: bool,
        kernel_backend_forward: KernelBackend | CutoTuneParameter,
        kernel_backend_backward: KernelBackend | CutoTuneParameter,
        BLOCK_SIZE_B_forward: int | CutoTuneParameter,
        BLOCK_SIZE_B_backward: int | CutoTuneParameter,
        BLOCK_SIZE_H_forward: int | CutoTuneParameter,
        BLOCK_SIZE_H_backward: int | CutoTuneParameter,
    ) -> torch.Tensor:
        output, rmsnorm_denominator = _forward(
            x=x,
            weight=weight,
            eps=eps,
            memory_efficient=memory_efficient,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        ctx.memory_efficient = memory_efficient
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.eps = eps
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        ctx.save_for_backward(x, weight, rmsnorm_denominator)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        memory_efficient = ctx.memory_efficient

        x, weight, rmsnorm_denominator = ctx.saved_tensors

        x_grad, weight_grad = _backward(
            x=x,
            weight=weight,
            eps=ctx.eps,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            memory_efficient=memory_efficient,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        return x_grad, weight_grad, *[None] * 8


def rmsnorm_cute(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    memory_efficient: bool = False,
    kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_Cute.apply(
        x,
        weight,
        eps,
        memory_efficient,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_H_backward,
    )

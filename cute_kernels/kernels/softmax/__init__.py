import torch

from ...cutotune import CutoTuneParameter
from .torch_implementation import softmax_torch
from .triton_implementation import softmax_backward_triton, softmax_forward_triton


class _Softmax_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        kernel_backend_forward: str,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        kernel_backend_backward: str,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        if x.size(-1) == 1:
            return torch.ones_like(x)

        x = x.contiguous()
        ctx.save_for_backward(x)

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        output = torch.empty_like(x)

        if kernel_backend_forward == "triton":
            softmax_forward_triton(
                x=x, output=output, BLOCK_SIZE_B=BLOCK_SIZE_B_forward, BLOCK_SIZE_H=BLOCK_SIZE_H_forward
            )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend_forward})")

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(output)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        if output_grad.size(-1) == 1:
            x_grad = torch.zeros_like(output_grad)
        else:
            output_grad = output_grad.contiguous()
            output = ctx.saved_tensors[0]

            x_grad = torch.empty_like(output)
            kernel_backend_backward = ctx.kernel_backend_backward

            if kernel_backend_backward == "triton":
                softmax_backward_triton(
                    output=output,
                    output_grad=output_grad,
                    x_grad=x_grad,
                    BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
                    BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
                )
            else:
                raise ValueError(f"unexpected kernel_backend ({kernel_backend_backward})")

        return x_grad, *[None] * 8


def softmax_cute(
    x: torch.Tensor,
    kernel_backend_forward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Softmax_Cute.apply(
        x,
        kernel_backend_forward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_H_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
    )

import torch
import triton

from .kernels import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


class _Swiglu_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        output = torch.empty_like(gate)

        ctx.save_for_backward(gate, up)

        original_shape = gate.size()
        gate = gate.view(-1)
        up = up.view(-1)

        assert gate.numel() == up.numel(), "both tensors should have same number of elements"
        assert gate.type() == up.type(), "both tensors should have same dtype"

        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        swiglu_forward_triton_kernel[grid](gate, up, output, num_elements, BLOCK_SIZE=1024)

        output = output.view(original_shape)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors

        original_shape = gate.size()
        gate = gate.view(-1)
        up = up.view(-1)

        num_elements = output_grad.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        # the kernel uses the gate and up tensors to store the gradients in-place for memory savings
        swiglu_backward_triton_kernel[grid](gate, up, output_grad, num_elements, BLOCK_SIZE=1024)

        gate = gate.view(original_shape)
        up = up.view(original_shape)

        return gate, up


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _Swiglu_Triton.apply(gate, up)

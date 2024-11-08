import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, ensure_same_strides, get_cartesian_product_cutotune_configs
from .cuda_implementation import (
    swiglu_backward_cuda_kernel,
    swiglu_backward_cuda_kernel_compileable,
    swiglu_forward_cuda_kernel,
    swiglu_forward_cuda_kernel_compileable,
)
from .torch_implementation import swiglu_torch
from .triton_implementation import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


class _Swiglu_KHD(torch.autograd.Function):
    @staticmethod
    @cutotune(
        configs=(
            get_cartesian_product_cutotune_configs(
                kernel_backend_forward=[KernelBackend.cuda, KernelBackend.triton],
                kernel_backend_backward=[KernelBackend.cuda, KernelBackend.triton],
                BLOCK_SIZE_forward=BLOCK_SIZES_POWERS_OF_2,
                BLOCK_SIZE_backward=BLOCK_SIZES_POWERS_OF_2,
            )
            if torch.cuda.is_available()
            else []
        ),
        triggers={"gate.dtype"},
    )
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
        kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
        BLOCK_SIZE_forward: int | CutoTuneParameter = CutoTuneParameter(),
        BLOCK_SIZE_backward: int | CutoTuneParameter = CutoTuneParameter(),
    ) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        gate, up = ensure_same_strides(gate, up)

        ctx.save_for_backward(gate, up)
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward
        ctx.kernel_backend_backward = kernel_backend_backward

        output = torch.empty_like(gate)

        if kernel_backend_forward == KernelBackend.cuda:
            assert gate.is_cuda, "tensor gate is not on GPU"
            assert up.is_cuda, "tensor up is not on GPU"

            if torch.compiler.is_compiling():
                swiglu_forward_cuda_kernel_compileable(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE_forward)
            else:
                swiglu_forward_cuda_kernel(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE_forward)
        elif kernel_backend_forward == KernelBackend.triton:
            num_elements = gate.numel()
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

            with torch.device(gate.device):
                swiglu_forward_triton_kernel[grid](
                    gate_ptr=gate,
                    up_ptr=up,
                    output_ptr=output,
                    num_elements=num_elements,
                    BLOCK_SIZE=BLOCK_SIZE_forward,
                )
        else:
            raise ValueError(f"unexpected kernel_backend_forward ({kernel_backend_forward})")

        return output

    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return _Swiglu_KHD._backward(
            output_grad=output_grad,
            gate=ctx.gate,
            up=ctx.up,
            kernel_backend_backward=ctx.kernel_backend_backward,
            BLOCK_SIZE_backward=ctx.BLOCK_SIZE_backward,
        )

    @staticmethod
    @cutotune(
        configs=(
            get_cartesian_product_cutotune_configs(
                kernel_backend_backward=[KernelBackend.cuda, KernelBackend.triton],
                BLOCK_SIZE_backward=BLOCK_SIZES_POWERS_OF_2,
            )
            if torch.cuda.is_available()
            else []
        ),
        triggers={"gate.dtype"},
    )
    def _backward(
        output_grad: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend_backward: KernelBackend | CutoTuneParameter,
        BLOCK_SIZE_backward: int | CutoTuneParameter,
    ) -> tuple[torch.Tensor | None]:
        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        if kernel_backend_backward == KernelBackend.cuda:
            if torch.compiler.is_compiling():
                swiglu_backward_cuda_kernel_compileable(
                    gate=gate,
                    up=up,
                    output_grad=output_grad,
                    gate_grad=gate_grad,
                    up_grad=up_grad,
                    BLOCK_SIZE=BLOCK_SIZE_backward,
                )
            else:
                swiglu_backward_cuda_kernel(
                    gate=gate,
                    up=up,
                    output_grad=output_grad,
                    gate_grad=gate_grad,
                    up_grad=up_grad,
                    BLOCK_SIZE=BLOCK_SIZE_backward,
                )
        elif kernel_backend_backward == KernelBackend.triton:
            num_elements = gate.numel()
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

            with torch.device(gate.device):
                swiglu_backward_triton_kernel[grid](
                    gate_ptr=gate,
                    up_ptr=up,
                    output_grad_ptr=output_grad,
                    gate_grad_ptr=gate_grad,
                    up_grad_ptr=up_grad,
                    num_elements=num_elements,
                    BLOCK_SIZE=BLOCK_SIZE_backward,
                )
        else:
            raise ValueError(f"unexpected kernel_backend_backward ({kernel_backend_backward})")

        return gate_grad, up_grad, None, None, None, None


def swiglu_khd(
    gate: torch.Tensor,
    up: torch.Tensor,
    kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_forward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_backward: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _Swiglu_KHD.apply(
        gate, up, kernel_backend_forward, kernel_backend_backward, BLOCK_SIZE_forward, BLOCK_SIZE_backward
    )

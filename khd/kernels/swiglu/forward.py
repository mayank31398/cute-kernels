import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, ensure_same_strides, get_cartesian_product_cutotune_configs
from .cuda_implementation import swiglu_forward_cuda_kernel, swiglu_forward_cuda_kernel_compileable
from .triton_implementation import swiglu_forward_triton_kernel


@cutotune(
    configs=(
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[1, 2, 4],
            BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
        )
        if torch.cuda.is_available()
        else []
    )
    + (
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[8],
            BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
            condition=lambda **kwargs: kwargs["gate"].dtype in [torch.float16, torch.bfloat16],
        )
        if torch.cuda.is_available()
        else []
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        vector_instruction_width=[None],
        BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
    ),
    triggers={"gate.dtype"},
)
def _forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    kernel_backend: KernelBackend | CutoTuneParameter,
    vector_instruction_width: int | CutoTuneParameter,
    BLOCK_SIZE: int | CutoTuneParameter,
) -> torch.Tensor:
    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    gate, up = ensure_same_strides(gate, up)
    output = torch.empty_like(gate)

    if kernel_backend == KernelBackend.cuda:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        if torch.compiler.is_compiling():
            swiglu_forward_cuda_kernel_compileable(
                gate=gate,
                up=up,
                output=output,
                vector_instruction_width=vector_instruction_width,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            swiglu_forward_cuda_kernel(
                gate=gate,
                up=up,
                output=output,
                vector_instruction_width=vector_instruction_width,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    elif kernel_backend == KernelBackend.triton:
        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        with torch.device(gate.device):
            swiglu_forward_triton_kernel[grid](
                gate_ptr=gate,
                up_ptr=up,
                output_ptr=output,
                num_elements=num_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output

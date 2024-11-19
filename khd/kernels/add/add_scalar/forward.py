import torch
import triton

from ....constants import CUDA_BLOCK_SIZES_POWERS_OF_2, TRITON_BLOCK_SIZES_POWERS_OF_2
from ....enums import KernelBackend
from ....utils import ceil_divide, cutotune, get_cartesian_product_cutotune_configs
from .cuda_implementation import add_scalar_forward_cuda_kernel, add_scalar_forward_cuda_kernel_compileable
from .triton_implementation import add_scalar_forward_triton_kernel


@cutotune(
    configs=(
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[1, 2, 4],
            BLOCK_SIZE=CUDA_BLOCK_SIZES_POWERS_OF_2,
        )
        if torch.cuda.is_available()
        else []
    )
    + (
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[8],
            BLOCK_SIZE=CUDA_BLOCK_SIZES_POWERS_OF_2,
            condition=lambda **kwargs: kwargs["x"].dtype in [torch.float16, torch.bfloat16],
        )
        if torch.cuda.is_available()
        else []
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        vector_instruction_width=[None],
        BLOCK_SIZE=TRITON_BLOCK_SIZES_POWERS_OF_2,
    ),
    triggers={"x.dtype"},
)
def _forward(
    x: torch.Tensor,
    y: float,
    kernel_backend: KernelBackend,
    vector_instruction_width: int,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    if y == 0:
        return x

    output = torch.empty_like(x)

    if kernel_backend == KernelBackend.cuda:
        assert x.is_cuda, "tensor x is not on GPU"

        if torch.compiler.is_compiling():
            add_scalar_forward_cuda_kernel_compileable(
                x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            add_scalar_forward_cuda_kernel(
                x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE
            )
    elif kernel_backend == KernelBackend.triton:
        assert vector_instruction_width is None

        num_elements = x.numel()

        with torch.device(x.device):
            add_scalar_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
                x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output

import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, get_cartesian_product_cutotune_configs
from .torch_implementation import rmsnorm_torch
from .triton_implementation import rmsnorm_forward_triton_kernel


class _RMSNorm_KHD(torch.autograd.Function):
    @staticmethod
    # @cutotune(
    #     configs=get_cartesian_product_cutotune_configs(
    #         kernel_backend=[KernelBackend.triton],
    #         BLOCK_SIZE_B=BLOCK_SIZES_POWERS_OF_2,
    #         BLOCK_SIZE_H=BLOCK_SIZES_POWERS_OF_2,
    #     ),
    #     triggers={"x.dtype"},
    # )
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        kernel_backend: KernelBackend | CutoTuneParameter,
        BLOCK_SIZE_B: int | CutoTuneParameter,
        BLOCK_SIZE_H: int | CutoTuneParameter,
    ) -> torch.Tensor:
        if x.stride(-1) != 1:
            x = x.contiguous()

        assert x.dim() > 1, "x should have more than 1 dimensions"

        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"

            weight = weight.contiguous()

        hidden_size = x.size(-1)
        num_elements = x.numel() // hidden_size

        output = torch.empty_like(x)

        x_view = x.view(-1, hidden_size)
        output_view = output.view(-1, hidden_size)

        if kernel_backend == KernelBackend.triton:
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE_B"]),)

            with torch.device(x.device):
                rmsnorm_forward_triton_kernel[grid](
                    x_ptr=x_view,
                    x_stride_b=x_view.stride(0),
                    x_stride_h=x_view.stride(1),
                    has_weight=weight is not None,
                    weight_ptr=weight,
                    output_ptr=output_view,
                    output_stride_b=output_view.stride(0),
                    output_stride_h=output_view.stride(1),
                    eps=eps,
                    B=num_elements,
                    H=hidden_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_H=BLOCK_SIZE_H,
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output


def rmsnorm_khd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_KHD.apply(x, weight, eps, kernel_backend, BLOCK_SIZE_B, BLOCK_SIZE_H)

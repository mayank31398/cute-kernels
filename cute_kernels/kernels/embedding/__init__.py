import torch

from ...enums import KernelBackend
from ...utils import ceil_divide, ensure_contiguous
from .torch_implementation import embedding_torch
from .triton_implementation import embedding_forward_triton_kernel


class _Embedding_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input_ids: torch.Tensor,
        wte: torch.Tensor,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_H: int,
    ) -> torch.Tensor:
        num_elements = input_ids.numel()
        hidden_size = wte.size(-1)

        output = torch.empty(num_elements, hidden_size, dtype=wte.dtype, device=input_ids.device)

        if kernel_backend == KernelBackend.triton:
            with torch.device(input_ids.device):
                embedding_forward_triton_kernel[
                    (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
                ](
                    x_ptr=input_ids,
                    wte_ptr=wte,
                    wte_stride_v=wte.stride(0),
                    wte_stride_h=wte.stride(1),
                    output_ptr=output,
                    output_stride_b=output.stride(0),
                    output_stride_h=output.stride(1),
                    B=num_elements,
                    H=hidden_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_H=BLOCK_SIZE_H,
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output.view(*input_ids.size(), hidden_size)


def embedding_cute(
    input_ids: torch.Tensor, wte: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> torch.Tensor:
    return _Embedding_Cute.apply(input_ids, wte, kernel_backend, BLOCK_SIZE_B, BLOCK_SIZE_H)

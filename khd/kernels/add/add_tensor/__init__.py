import torch

from ....enums import KernelBackend
from ....utils import Config, CutoTune, ensure_same_strides
from .cuda_implementation import add_tensor_forward_cuda
from .triton_implementation import add_tensor_forward_triton


class _AddTensor(torch.autograd.Function):
    @staticmethod
    @CutoTune(
        configs=[Config({"kernel_backend": KernelBackend.cuda}), Config({"kernel_backend": KernelBackend.triton})],
        triggers={"x.dtype"},
    )
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_backend: KernelBackend,
        vectorized_loop_size: int | None = None,
        BLOCK_SIZE: int | None = None,
    ) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert y.is_cuda, "tensor y is not on GPU"

        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y, expected_stride=x.stride())
        output = torch.empty_like(x)

        kwargs = {"x": x, "y": y, "output": output}
        if BLOCK_SIZE is not None:
            kwargs["BLOCK_SIZE"] = BLOCK_SIZE

        if kernel_backend == KernelBackend.cuda:
            if vectorized_loop_size is not None:
                kwargs["vectorized_loop_size"] = vectorized_loop_size

            add_tensor_forward_cuda(**kwargs)
        elif kernel_backend == KernelBackend.triton:
            assert vectorized_loop_size is None

            add_tensor_forward_triton(**kwargs)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def add_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _AddTensor.apply(x, y)

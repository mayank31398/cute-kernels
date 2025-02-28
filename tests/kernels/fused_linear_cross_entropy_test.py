import random
from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import fused_linear_cross_entropy_cute, fused_linear_cross_entropy_torch

from ..test_commons import TestCommons


_SEED = 42


class FusedLinearCrossEntropyTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            ["triton"],  # kernel_backend_backward
            [
                fused_linear_cross_entropy_cute,
                torch.compile(fused_linear_cross_entropy_cute, fullgraph=True),
            ],  # function
        )
    )
    def test_fused_linear_cross_entropy(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend_backward: str,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)

        vocab_size = random.randint(max(100, size[0] - 100), size[0] + 100)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (vocab_size, size[1]), device=device, dtype=dtype, std=2e-3
        )
        logits_multiplier = 0.7

        labels = torch.randint(0, vocab_size, (x_kernel.size(0),), device=x_kernel.device)

        loss_kernel = function(
            x=x_kernel,
            weight=weight_kernel,
            labels=labels,
            logits_multiplier=logits_multiplier,
            kernel_backend_backward=kernel_backend_backward,
        )
        loss_expected = fused_linear_cross_entropy_torch(
            x=x_expected, weight=weight_expected, labels=labels, logits_multiplier=logits_multiplier
        )

        loss_kernel.backward()
        loss_expected.backward()

        self.assert_equal_tensors(loss_kernel, loss_expected, False, atol_float32=2e-4, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
        self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False)

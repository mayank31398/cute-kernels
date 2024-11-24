from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import rmsnorm_cute, rmsnorm_torch

from ..test_commons import TestCommons


_EPSILON = 1e-5
_SEED = 4


class RMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(400),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [True],  # memory_efficient
            [rmsnorm_cute, torch.compile(rmsnorm_cute)],  # function
        )
    )
    def test_rmsnorm(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        memory_efficient: bool,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)

        z_kernel = function(x=x_kernel, weight=weight_kernel, eps=_EPSILON, memory_efficient=memory_efficient)
        z_expected = rmsnorm_torch(x=x_expected, weight=weight_expected, eps=_EPSILON)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float16=8e-3, rtol_float16=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float16=0.07, rtol_float16=0)
        # self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False, atol_float32=8.5e-5, rtol_float32=0)

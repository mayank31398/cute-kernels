from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import linear_cute, linear_torch

from ..test_commons import TestCommons


_EPSILON = 1e-5
_SEED = 42


class LinearTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes()[:1],  # dtype
            [False, True],  # has_bias
            [linear_cute, torch.compile(linear_cute, fullgraph=True)][:1],  # function
        )
    )
    def test_linear(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        has_bias: bool,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        input_kernel, input_expected = self.get_random_duplicated_tensors((400, size[-1]), device=device, dtype=dtype)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        bias_kernel = None
        bias_expected = None
        if has_bias:
            bias_kernel, bias_expected = self.get_random_duplicated_tensors(size[0], device=device, dtype=dtype)

        z_kernel = function(input=input_kernel, weight=weight_kernel, bias=bias_kernel)
        z_expected = linear_torch(input=input_expected, weight=weight_expected, bias=bias_expected)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=4e-3, rtol_float32=1e-4)

        # print(size, dtype)
        # assert False
        # self.assert_equal_tensors(input_kernel.grad, input_expected.grad, False, atol_float16=9e-2, rtol_float16=0)
        # self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False, atol_float16=9e-2, rtol_float16=0)

        # if has_bias:
        #     self.assert_equal_tensors(
        #         bias_kernel.grad,
        #         bias_expected.grad,
        #         False,
        #         atol_float32=6.5e-5,
        #         rtol_float32=0,
        #         atol_float16=0.1,
        #         rtol_float16=0.01,
        #     )

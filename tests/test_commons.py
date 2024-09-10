import random
from itertools import product
from typing import Any
from unittest import TestCase

import torch
import torch.nn as nn
from torch.testing import assert_close


class TestCommons(TestCase):
    @staticmethod
    def get_all_devices() -> list[torch.device]:
        return [torch.device("cpu"), torch.device("cuda")]

    @staticmethod
    def get_dtypes() -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @staticmethod
    def get_1d_tensor_sizes() -> list[tuple[int]]:
        sizes = set()
        # powers of 2
        for i in range(15):
            start = 2**i
            for j in range(10):
                sizes.add(start + j)
        # not powers of 2
        for _ in range(50):
            sizes.add(3000 + random.randint(-1000, 1000))
        return sizes

    @staticmethod
    def get_2d_tensor_sizes() -> list[tuple[int]]:
        sizes = set()
        # powers of 2
        for i in range(15):
            start = 2**i
            for j in range(10):
                sizes.add((start + j, start + j))
        # not powers of 2
        for _ in range(50):
            sizes.add((3000 + random.randint(-1000, 1000), 3000 + random.randint(-1000, 1000)))
        return sizes

    def make_args_matrix(*args_lists) -> list[Any]:
        return [p for p in product(*args_lists)]

    def assert_equal_tensors(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        exact_match: bool,
        rtol_float32: float = None,
        atol_float32: float = None,
        rtol_float16: float = None,
        atol_float16: float = None,
        rtol_bfloat16: float = None,
        atol_bfloat16: float = None,
    ) -> None:
        if exact_match:
            assert x.equal(y)
        else:
            assert x.dtype == y.dtype
            dtype = x.dtype

            if dtype == torch.float32:
                assert_close(x, y, rtol=rtol_float32, atol=atol_float32)
            elif dtype == torch.float16:
                assert_close(x, y, rtol=rtol_float16, atol=atol_float16)
            elif dtype == torch.bfloat16:
                assert_close(x, y, rtol=rtol_bfloat16, atol=atol_bfloat16)
            else:
                raise ValueError(f"unexpected dtype ({dtype})")

    def get_activation_function(self, is_glu: bool) -> nn.Module:
        return nn.GLU() if is_glu else nn.GELU(approximate="tanh")

    def get_random_duplicated_non_leaf_tensors(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor]:
        x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_()

        # make non leaf
        x = x + 1
        x_clone = x_clone + 1

        return x, x_clone

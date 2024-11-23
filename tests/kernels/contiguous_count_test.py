from typing import Callable

import torch
from parameterized import parameterized

from khd import KernelBackend, contiguous_count_khd

from ..test_commons import TestCommons


MAX_EXPERTS = 72


class ContiguousCountTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [KernelBackend.triton],  # kernel_backend
            [64],  # BLOCK_SIZE_B
            [contiguous_count_khd, torch.compile(contiguous_count_khd)],  # function
        )
    )
    def test_contiguous_count(
        self,
        size: tuple[int],
        device: torch.device,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_B: int,
        function: Callable,
    ) -> None:
        x = torch.randint(0, MAX_EXPERTS, size, device=device, dtype=torch.long)

        z_kernel = function(x=x, start=0, end=MAX_EXPERTS, kernel_backend=kernel_backend, BLOCK_SIZE_B=BLOCK_SIZE_B)
        z_expected = x.view(-1).bincount(minlength=MAX_EXPERTS)

        self.assert_equal_tensors(z_kernel, z_expected, True)
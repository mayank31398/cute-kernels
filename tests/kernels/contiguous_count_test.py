from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from cute_kernels import KernelBackend, contiguous_count_cute, contiguous_count_torch

from ..test_commons import TestCommons


_MAX_EXPERTS = 72
_SEED = 42


class ContiguousCountTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [1, 2, 4, 8],  # thread_block_cluster_size
            [KernelBackend.cuda],  # kernel_backend
            [64],  # BLOCK_SIZE_B
            [contiguous_count_cute, torch.compile(contiguous_count_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [None],  # thread_block_cluster_size
            [KernelBackend.triton],  # kernel_backend
            [64],  # BLOCK_SIZE_B
            [contiguous_count_cute, torch.compile(contiguous_count_cute, fullgraph=True)],  # function
        )
    )
    def test_contiguous_count(
        self,
        size: int,
        device: torch.device,
        thread_block_cluster_size: int,
        kernel_backend: KernelBackend,
        BLOCK_SIZE: int,
        function: Callable,
    ) -> None:
        set_seed(_SEED)
        x = torch.randint(0, _MAX_EXPERTS, (size,), device=device, dtype=torch.long)

        z_kernel = function(
            x=x,
            size=_MAX_EXPERTS,
            thread_block_cluster_size=thread_block_cluster_size,
            kernel_backend=kernel_backend,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        z_expected = contiguous_count_torch(x.view(-1), size=_MAX_EXPERTS)

        self.assert_equal_tensors(z_kernel, z_expected, True)

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, embedding_cute, embedding_torch

from ..test_commons import TestCommons


class EmbeddingTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(51, 17), (19, 239), (7, 7537), (9, 1749)],  # input_ids_size
            [(7153, 937), (27153, 1937), (97153, 2937), (17153, 31937)],  # weight_size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.triton],  # kernel_backend_forward
            [KernelBackend.triton],  # kernel_backend_backward
            [64],  # BLOCK_SIZE_B_forward
            [64],  # BLOCK_SIZE_B_backward
            [64],  # BLOCK_SIZE_H_forward
            [64],  # BLOCK_SIZE_H_backward
            [embedding_cute, torch.compile(embedding_cute)],  # function
        )
    )
    def test_embedding(
        self,
        input_ids_size: tuple[int],
        weight_size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_H_backward: int,
        function: Callable,
    ) -> None:
        vocab_size = weight_size[0] - 1
        input_ids = torch.randint(0, vocab_size, input_ids_size, device=device, dtype=torch.long)

        weight_kernel, weight_expected = self.get_random_duplicated_tensors(weight_size, device=device, dtype=dtype)

        z_kernel = function(
            input_ids,
            weight_kernel,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
            BLOCK_SIZE_B_forward=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_B_backward=BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H_forward=BLOCK_SIZE_H_forward,
            BLOCK_SIZE_H_backward=BLOCK_SIZE_H_backward,
        )
        z_expected = embedding_torch(input_ids, weight_expected)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False, atol_float32=1e-10, rtol_float32=0)

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import (
    COMMON_VECTOR_INSTRUCTION_WIDTHS,
    MAX_FP16_BF16_INSTRUCTION_WIDTH,
    KernelBackend,
    swiglu_cute,
    swiglu_torch,
)

from ..test_commons import TestCommons


class SwigluTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda],  # kernel_backend_forward
            [KernelBackend.cuda],  # kernel_backend_backward
            COMMON_VECTOR_INSTRUCTION_WIDTHS,  # vector_instruction_width_forward
            COMMON_VECTOR_INSTRUCTION_WIDTHS,  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.bfloat16, torch.float16],  # dtype
            [KernelBackend.cuda],  # kernel_backend_forward
            [KernelBackend.cuda],  # kernel_backend_backward
            [MAX_FP16_BF16_INSTRUCTION_WIDTH],  # vector_instruction_width_forward
            [MAX_FP16_BF16_INSTRUCTION_WIDTH],  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.triton],  # kernel_backend_forward
            [KernelBackend.triton],  # kernel_backend_backward
            [None],  # vector_instruction_width_forward
            [None],  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda],  # kernel_backend_forward
            [KernelBackend.triton],  # kernel_backend_backward
            COMMON_VECTOR_INSTRUCTION_WIDTHS,  # vector_instruction_width_forward
            [None],  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.bfloat16, torch.float16],  # dtype
            [KernelBackend.cuda],  # kernel_backend_forward
            [KernelBackend.triton],  # kernel_backend_backward
            [MAX_FP16_BF16_INSTRUCTION_WIDTH],  # vector_instruction_width_forward
            [None],  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.triton],  # kernel_backend_forward
            [KernelBackend.cuda],  # kernel_backend_backward
            [None],  # vector_instruction_width_forward
            COMMON_VECTOR_INSTRUCTION_WIDTHS,  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.bfloat16, torch.float16],  # dtype
            [KernelBackend.triton],  # kernel_backend_forward
            [KernelBackend.cuda],  # kernel_backend_backward
            [None],  # vector_instruction_width_forward
            [MAX_FP16_BF16_INSTRUCTION_WIDTH],  # vector_instruction_width_backward
            [1024],  # BLOCK_SIZE_forward
            [1024],  # BLOCK_SIZE_backward
            [swiglu_cute, torch.compile(swiglu_cute, fullgraph=True)],  # function
        )
    )
    def test_swiglu(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
        vector_instruction_width_forward: int,
        vector_instruction_width_backward: int,
        BLOCK_SIZE_forward: int,
        BLOCK_SIZE_backward: int,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(
            x_kernel,
            y_kernel,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
            vector_instruction_width_forward=vector_instruction_width_forward,
            vector_instruction_width_backward=vector_instruction_width_backward,
            BLOCK_SIZE_forward=BLOCK_SIZE_forward,
            BLOCK_SIZE_backward=BLOCK_SIZE_backward,
        )
        z_expected = swiglu_torch(x_expected, y_expected)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5.5e-6, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, False, atol_float32=5e-6, rtol_float32=0)

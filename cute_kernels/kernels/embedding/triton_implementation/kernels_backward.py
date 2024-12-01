import triton
import triton.language as tl


@triton.jit
def embedding_backward_triton_kernel(
    input_ids_ptr,
    output_grad_ptr,
    weight_grad_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    input_ids_ptrs = input_ids_ptr + indices_b
    input_ids = tl.load(input_ids_ptrs, mask=mask_b)

    output_grad_ptrs = output_grad_ptr + indices_b[:, None] * H + indices_h[None, :]
    output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

    weight_grad_ptrs = weight_grad_ptr + input_ids[:, None] * H + indices_h[None, :]
    tl.atomic_add(weight_grad_ptrs, output_grad, mask=mask_bh)

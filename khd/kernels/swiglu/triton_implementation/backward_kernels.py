import triton
import triton.language as tl


def _swiglu_backward(gate_ptr, up_ptr, output_grad_ptr, gate_grad_ptr, up_grad_ptr, indices, mask):
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)
    output_grad = tl.load(output_grad_ptr + indices, mask=mask)

    gate_sigmoid = tl.sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))
    up_grad = output_grad * gate_silu

    tl.store(gate_grad_ptr + indices, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices, up_grad, mask=mask)


@triton.jit
def swiglu_backward_triton_kernel(
    gate_ptr, up_ptr, output_grad_ptr, gate_grad_ptr, up_grad_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    is_last_block = pid == num_blocks - 1

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if is_last_block:
        mask = indices < num_elements
        _swiglu_backward(
            gate_ptr=gate_ptr,
            up_ptr=up_ptr,
            output_grad_ptr=output_grad_ptr,
            gate_grad_ptr=gate_grad_ptr,
            up_grad_ptr=up_grad_ptr,
            indices=indices,
            mask=mask,
        )
    else:
        _swiglu_backward(
            gate_ptr=gate_ptr,
            up_ptr=up_ptr,
            output_grad_ptr=output_grad_ptr,
            gate_grad_ptr=gate_grad_ptr,
            up_grad_ptr=up_grad_ptr,
            indices=indices,
            mask=None,
        )

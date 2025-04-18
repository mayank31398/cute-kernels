import triton
import triton.language as tl


@triton.jit
def _add_scalar(x_ptr, y, output_ptr, indices, mask):
    x = tl.load(x_ptr + indices, mask=mask)
    tl.store(output_ptr + indices, x + y, mask=mask)


@triton.jit
def _add_scalar_triton_kernel(x_ptr, y, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        _add_scalar(x_ptr=x_ptr, y=y, output_ptr=output_ptr, indices=indices, mask=None)
    else:
        _add_scalar(x_ptr=x_ptr, y=y, output_ptr=output_ptr, indices=indices, mask=indices < N)

import triton
import triton.language as tl


@triton.jit
def _contiguous_count_low_atomic_add_triton_kernel(
    x_ptr, output_ptr, B, C, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(B, num_programs)

    indices_c = tl.arange(0, BLOCK_SIZE_C)
    mask_c = indices_c < C

    program_start = pid * num_elements_per_program
    program_end = min(program_start + num_elements_per_program, B)
    num_elements_in_current_program = program_end - program_start

    if num_elements_in_current_program > 0:
        num_loops = tl.cdiv(num_elements_in_current_program, BLOCK_SIZE_B)
        counts = tl.zeros((BLOCK_SIZE_C,), dtype=tl.uint32)

        for i in range(num_loops):
            indices_b = program_start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            mask_b = indices_b < program_end

            x = tl.load(x_ptr + indices_b, mask=mask_b, other=-1)

            equal = (x[:, None] == indices_c[None, :]).to(tl.uint32)
            counts += tl.sum(equal, axis=0)

        tl.atomic_add(output_ptr + indices_c, counts, mask=mask_c)


@triton.jit
def _contiguous_count_high_atomic_add_triton_kernel(x_ptr, output_ptr, B, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < B

    x = tl.load(x_ptr + indices, mask=mask)
    tl.atomic_add(output_ptr + x, 1, mask=mask)

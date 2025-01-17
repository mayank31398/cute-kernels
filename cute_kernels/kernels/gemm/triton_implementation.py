import torch
import triton
import triton.language as tl

from ...constants import LIBRARY_NAME
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...math import ceil_divide, get_powers_of_2
from ...utils import cute_op, get_num_elements_and_hidden_size


_KERNEL_NAME = "gemm_triton"


@triton.jit
def _gemm_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    use_tf32: tl.constexpr,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # a -> M x K
    # b -> K x N
    # c -> M x N

    pid = tl.program_id(axis=0)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_programs_n
    pid_n = pid % num_programs_n

    indices_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = indices_m < M
    mask_n = indices_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        indices_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = indices_k < K

        mask_mk = mask_m[:, None] & mask_k[None, :]
        mask_kn = mask_k[:, None] & mask_n[None, :]

        a_ptrs = a_ptr + indices_m[:, None] * K + indices_k[None, :]
        a = tl.load(a_ptrs, mask=mask_mk, other=0)

        b_ptrs = b_ptr + indices_k[:, None] * N + indices_n[None, :]
        b = tl.load(b_ptrs, mask=mask_kn, other=0)

        accumulator = tl.dot(a, b, accumulator, allow_tf32=use_tf32)

    c_ptrs = c_ptr + indices_m[:, None] * N + indices_n[None, :]
    tl.store(c_ptrs, accumulator.to(a_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _condition(input: torch.Tensor, BLOCK_SIZE_M: int, BLOCK_SIZE_K: int, BLOCK_SIZE_N: int, **kwargs) -> bool:
    if input.dtype == torch.float32 and BLOCK_SIZE_M == 128 and BLOCK_SIZE_K == 128 and BLOCK_SIZE_N == 128:
        return False

    return True


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE_M=get_powers_of_2(16, 128),
        BLOCK_SIZE_K=get_powers_of_2(16, 128),
        BLOCK_SIZE_N=get_powers_of_2(16, 128),
        num_warps=get_powers_of_2(1, 32),
        num_stages=[1, 2],
        condition=_condition,
    ),
    default_config=CutoTuneConfig(
        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_K=64, BLOCK_SIZE_N=128, num_warps=8, num_stages=2)
    ),
    triggers={"a.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"c"})
def gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    use_tf32: bool,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_K: int,
    BLOCK_SIZE_N: int,
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = get_num_elements_and_hidden_size(a)
    N = b.size(-1)

    with torch.device(input.device):
        _gemm_triton_kernel[(ceil_divide(M, BLOCK_SIZE_M) * ceil_divide(N, BLOCK_SIZE_N),)](
            a_ptr=a,
            b_ptr=b,
            c_ptr=c,
            use_tf32=use_tf32,
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

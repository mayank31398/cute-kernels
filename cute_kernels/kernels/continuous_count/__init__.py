import torch

from ...constants import COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2, DTYPE_TO_SIZE, THREAD_BLOCK_CLUSTER_SIZES
from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune, get_cartesian_product_cutotune_configs
from ...math import get_next_power_of_2
from ...utils import get_sm_count
from .cuda_implementation import continuous_count_cuda
from .torch_implementation import continuous_count_torch
from .triton_implementation import continuous_count_triton


@cutotune(
    get_cartesian_product_cutotune_configs(
        kernel_backend=["triton"],
        thread_block_cluster_size=[None],
        BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
        condition=lambda **kwargs: kwargs["size"] <= 256,
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=["cuda"],
        thread_block_cluster_size=THREAD_BLOCK_CLUSTER_SIZES,
        BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    ),
    default_config=CutoTuneConfig(dict(kernel_backend="cuda", thread_block_cluster_size=8, BLOCK_SIZE=1024)),
    functional_triggers={
        "next_power_of_2(size)": lambda **kwargs: get_next_power_of_2(kwargs["size"]),
        "sizeof(dtype)": lambda **kwargs: DTYPE_TO_SIZE[kwargs["x"].dtype],
    },
)
def _continuous_count_cute(
    x: torch.Tensor,
    size: int,
    thread_block_cluster_size: int,
    kernel_backend: str,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    if kernel_backend == "cuda":
        output = torch.empty(size, dtype=torch.uint32, device=x.device)
        continuous_count_cuda(
            x=x,
            output=output,
            sm_count=get_sm_count(x.device),
            thread_block_cluster_size=thread_block_cluster_size,
            size=size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif kernel_backend == "triton":
        assert thread_block_cluster_size is None
        output = torch.zeros(size, dtype=torch.uint32, device=x.device)

        continuous_count_triton(
            x=x, output=output, size=size, BLOCK_SIZE=BLOCK_SIZE, BLOCK_SIZE_C=get_next_power_of_2(size)
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


@torch.no_grad()
def continuous_count_cute(
    x: torch.Tensor,
    size: int,
    thread_block_cluster_size: int = CutoTuneParameter(),
    kernel_backend: str = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    if size == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    return _continuous_count_cute(
        x=x,
        size=size,
        thread_block_cluster_size=thread_block_cluster_size,
        kernel_backend=kernel_backend,
        BLOCK_SIZE=BLOCK_SIZE,
    )

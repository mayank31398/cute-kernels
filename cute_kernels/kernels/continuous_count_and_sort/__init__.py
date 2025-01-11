import torch

from ...constants import COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2, DTYPE_TO_SIZE, THREAD_BLOCK_CLUSTER_SIZES
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...math import get_next_power_of_2
from ...utils import get_sm_count
from .cuda_implementation import continuous_count_and_sort_cuda
from .torch_implementation import continuous_count_and_sort_torch


@torch.no_grad()
@cutotune(
    get_cartesian_product_cutotune_configs(
        thread_block_cluster_size=THREAD_BLOCK_CLUSTER_SIZES, BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2
    ),
    default_config=CutoTuneConfig(dict(thread_block_cluster_size=8, BLOCK_SIZE=1024)),
    functional_triggers={
        "next_power_of_2(size)": lambda **kwargs: get_next_power_of_2(kwargs["size"]),
        "sizeof(dtype)": lambda **kwargs: DTYPE_TO_SIZE[kwargs["x"].dtype],
    },
)
def continuous_count_and_sort_cute(
    x: torch.Tensor,
    size: int,
    thread_block_cluster_size: int,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor]:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    count_output = torch.empty(size, dtype=torch.uint32, device=x.device)
    sorted_output = torch.empty_like(x, dtype=torch.uint32)
    argsort_output = torch.empty_like(x, dtype=torch.uint32)

    continuous_count_and_sort_cuda(
        x=x,
        count_output=count_output,
        sorted_output=sorted_output,
        argsort_output=argsort_output,
        sm_count=get_sm_count(x.device),
        thread_block_cluster_size=thread_block_cluster_size,
        size=size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return count_output, sorted_output, argsort_output

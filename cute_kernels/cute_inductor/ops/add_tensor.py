import torch

from ...kernels import add_tensor_cute, add_tensor_torch
from ..config import ReplacementConfig


def _get_example_inputs() -> tuple[torch.Tensor]:
    return torch.empty(8, 8)


add_tensor_replacement_config = ReplacementConfig(
    pattern_function=add_tensor_torch,
    replacement_function=add_tensor_cute,
    example_inputs_functions=_get_example_inputs,
)

import inspect
from functools import partial

import torch

from ...kernels import rmsnorm_cute, rmsnorm_torch
from ..config import ReplacementConfig
from ..utils import parse_args_and_kwargs_to_kwargs


def _get_example_inputs() -> list[dict]:
    return [
        {
            "x": torch.randn(4, 4, device=torch.cuda.current_device()),
            "weight": torch.randn(4, device=torch.cuda.current_device()),
            "eps": None,
        },
        {
            "x": torch.randn(4, 4, device=torch.cuda.current_device()),
            "weight": None,
            "eps": None,
        },
    ]


rmsnorm_replacement_config = ReplacementConfig(
    search_function=rmsnorm_torch,
    replacement_function=rmsnorm_cute,
    example_inputs_function=_get_example_inputs,
    prepare_inputs_function=partial(
        parse_args_and_kwargs_to_kwargs, signature=inspect.getfullargspec(rmsnorm_torch).args
    ),
)

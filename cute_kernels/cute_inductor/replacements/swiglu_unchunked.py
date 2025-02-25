import torch

from ...kernels import swiglu_unchunked_cute, swiglu_unchunked_torch
from ..config import ReplacementConfig
from ..utils import parse_args_and_kwargs_to_kwargs


def _prepare_inputs_function(args: list, kwargs: dict) -> dict:
    kwargs = parse_args_and_kwargs_to_kwargs(["input", "chunks", "dim"], args, kwargs)

    kwargs["x"] = kwargs.pop("input")
    kwargs.pop("chunks")
    kwargs.pop("dim")

    return kwargs


def _get_example_inputs() -> list[dict]:
    return [{"x": torch.randn(4, 4, device=torch.cuda.current_device())}]


swiglu_unchunked_replacement_config = ReplacementConfig(
    search_function=swiglu_unchunked_torch,
    replacement_function=swiglu_unchunked_cute,
    example_inputs_function=_get_example_inputs,
    prepare_inputs_function=_prepare_inputs_function,
)

import torch

from cute_kernels import (
    CuteInductor,
    ReplacementConfig,
    parse_args_and_kwargs_to_kwargs,
    swiglu_unchunked_cute,
    swiglu_unchunked_torch,
)


def f(x):
    x = x * 4
    x = x + 3
    x = swiglu_unchunked_torch(x)
    x = x - 3
    return x


def prepare_inputs_function(args: list, kwargs: dict) -> dict:
    kwargs = parse_args_and_kwargs_to_kwargs(["input", "chunks", "dim"], args, kwargs)

    kwargs["x"] = kwargs.pop("input")
    kwargs.pop("chunks")
    kwargs.pop("dim")

    return kwargs


device = torch.cuda.current_device()


compiled_f = torch.compile(
    f,
    backend=CuteInductor(
        replacement_configs=[
            ReplacementConfig(
                search_function=swiglu_unchunked_torch,
                replacement_function=swiglu_unchunked_cute,
                example_inputs=torch.randn(8, 8, device=device),
                prepare_inputs_function=prepare_inputs_function,
            )
        ]
    ).compiler,
)(torch.randn(8, 8, device=device))

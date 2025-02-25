from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
from torch._dynamo import lookup_backend
from torch.fx import replace_pattern

from cute_kernels import parse_args_and_kwargs_to_kwargs, swiglu_unchunked_cute, swiglu_unchunked_torch


def f(x):
    x = x * 4
    x = x + 3
    x = swiglu_unchunked_torch(x)
    x = x - 3
    return x


class GraphCapture:
    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        self.gm = gm
        return gm.forward


def prepare_inputs_function(args: list, kwargs: dict) -> dict:
    kwargs = parse_args_and_kwargs_to_kwargs(["input", "chunks", "dim"], args, kwargs)

    kwargs["x"] = kwargs.pop("input")
    kwargs.pop("chunks")
    kwargs.pop("dim")

    return kwargs


@dataclass
class ReplacementConfig:
    search_function: Callable
    replacement_function: Callable
    example_inputs: tuple[torch.Tensor]
    prepare_inputs_function: Callable


class CuteInductor:
    def __init__(
        self, replacement_configs: list[ReplacementConfig], use_torch_inductor_after_cute_inductor: bool = True
    ) -> None:
        self.replacement_configs = deepcopy(replacement_configs)

        graph_capture = GraphCapture()

        for replacement_config in self.replacement_configs:
            example_inputs = replacement_config.example_inputs
            example_inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)

            torch.compile(replacement_config.search_function, backend=graph_capture.compiler)(*example_inputs)
            replacement_config.search_graph = graph_capture.gm.graph

            torch.compile(replacement_config.replacement_function, backend=graph_capture.compiler)(*example_inputs)
            replacement_config.replacement_graph = graph_capture.gm.graph

        self.use_torch_inductor_after_cute_inductor = use_torch_inductor_after_cute_inductor

    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        print("-" * 50 + "\ngraph before cute inductor\n" + "-" * 50)
        gm.print_readable()

        for replacement_config in self.replacement_configs:
            replace_pattern(gm, replacement_config.search_graph, replacement_config.replacement_graph)

        print("-" * 50 + "\ngraph after cute inductor\n" + "-" * 50)
        gm.print_readable()

        if self.use_torch_inductor_after_cute_inductor:
            inductor = lookup_backend("inductor")
            compiled = inductor(gm, example_inputs)
        else:
            compiled = gm.forward

        return compiled


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

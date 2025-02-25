from copy import deepcopy
from typing import Callable

import torch
from torch._dynamo import lookup_backend
from torch.fx import replace_pattern

from ..utils import enable_cute_tracing, get_boolean_env_variable
from .config import ReplacementConfig


_DEBUG_CUTEINDUCTOR = get_boolean_env_variable("DEBUG_CUTEINDUCTOR", True)


class _GraphCaptureDummyCompiler:
    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        self.gm = gm
        return gm.forward


class CuteInductor:
    def __init__(
        self, replacement_configs: list[ReplacementConfig], use_torch_inductor_after_cute_inductor: bool = True
    ) -> None:
        self.replacement_configs = deepcopy(replacement_configs)

        graph_capture = _GraphCaptureDummyCompiler()

        for replacement_config in self.replacement_configs:
            example_inputs = replacement_config.example_inputs_function()

            torch.compile(replacement_config.search_function, backend=graph_capture.compiler)(*example_inputs)
            replacement_config.search_graph = graph_capture.gm.graph

            torch.compile(replacement_config.replacement_function, backend=graph_capture.compiler)(*example_inputs)
            replacement_config.replacement_graph = graph_capture.gm.graph

        self.use_torch_inductor_after_cute_inductor = use_torch_inductor_after_cute_inductor

    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        with enable_cute_tracing():
            if _DEBUG_CUTEINDUCTOR:
                print("-" * 50 + "\ngraph before cute inductor\n" + "-" * 50)
                gm.print_readable()

            for replacement_config in self.replacement_configs:
                replace_pattern(gm, replacement_config.search_graph, replacement_config.replacement_graph)

            if _DEBUG_CUTEINDUCTOR:
                print("-" * 50 + "\ngraph after cute inductor\n" + "-" * 50)
                gm.print_readable()

            if self.use_torch_inductor_after_cute_inductor:
                inductor = lookup_backend("inductor")
                compiled = inductor(gm, example_inputs)
            else:
                compiled = gm.forward

            return compiled

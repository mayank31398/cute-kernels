from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch._dynamo import lookup_backend
from torch.fx import Node


def search(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return x[0] * F.silu(x[1])


def replace(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return F.relu(x[0]) + F.relu(-x[1])


def f(x):
    x = x * 4
    x = x + 3
    x = search(x)
    x = x - 3
    return x


def parse_args_and_kwargs_to_kwargs(signature: list[str], args: list, kwargs: dict) -> dict:
    result = {}
    for key, value in zip(signature, args):
        result[key] = value

    for key, value in kwargs.items():
        result[key] = value

    return result


class GraphCapture:
    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        self.gm = gm
        return gm.forward


@dataclass
class ReplacementConfig:
    search_function: Callable
    replacement_function: Callable
    example_inputs: tuple[torch.Tensor]


class CuteInductor:
    def __init__(
        self, replacement_configs: list[ReplacementConfig], use_torch_inductor_after_cute_inductor: bool = True
    ) -> None:
        self.replacement_configs = deepcopy(replacement_configs)

        for replacement_config in self.replacement_configs:
            example_inputs = replacement_config.example_inputs
            example_inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)

            graph_capture = GraphCapture()
            torch.compile(replacement_config.search_function, backend=graph_capture.compiler)(*example_inputs)
            replacement_config.graph = graph_capture.gm.graph

        self.use_torch_inductor_after_cute_inductor = use_torch_inductor_after_cute_inductor

    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        print("-" * 50 + "\ngraph before cute inductor\n" + "-" * 50)
        gm.print_readable()

        for graph_node in gm.graph.nodes:
            if graph_node.op == "placeholder":
                continue

            for replacement_config in self.replacement_configs:
                match, output_nodes = self._match_subgraph(graph_node, next(iter(replacement_config.graph.nodes)))

                if match:
                    with gm.graph.inserting_after(graph_node):
                        kwargs = parse_args_and_kwargs_to_kwargs(
                            ["input", "chunks", "dim"], graph_node.args, graph_node.kwargs
                        )

                        kwargs["x"] = kwargs.pop("input")
                        kwargs.pop("chunks")
                        kwargs.pop("dim")

                        new_node = gm.graph.call_function(replacement_config.replacement_function, kwargs=kwargs)

                    print("replacing with rmsnorm_cute")

                    graph_node.replace_all_uses_with(new_node)
                    for output_node in output_nodes:
                        output_node.replace_all_uses_with(new_node)

                    gm.graph.erase_node(graph_node)
                    gm.graph.eliminate_dead_code()

        print("-" * 50 + "\ngraph after cute inductor\n" + "-" * 50)
        gm.print_readable()

        if self.use_torch_inductor_after_cute_inductor:
            inductor = lookup_backend("inductor")
            compiled = inductor(gm, example_inputs)
        else:
            compiled = gm.forward

        return compiled

    def _match_subgraph(self, graph_node: Node, search_node: Node) -> tuple[bool, list[Node]]:
        child_matches = []
        output_nodes = []

        if search_node.op == "output":
            return True, [graph_node.prev]

        if search_node.op == "placeholder":
            for sn in search_node.users.keys():
                child_match, _output_nodes = self._match_subgraph(graph_node, sn)

                child_matches.append(child_match)
                output_nodes.extend(_output_nodes)

            return all(child_matches), output_nodes

        if (
            graph_node.op != search_node.op
            or graph_node.target != search_node.target
            or len(graph_node.users) != len(search_node.users)
        ):
            return False, []

        for gn, sn in zip(graph_node.users, search_node.users):
            child_match, _output_nodes = self._match_subgraph(gn, sn)

            child_matches.append(child_match)
            output_nodes.extend(_output_nodes)

        return all(child_matches), output_nodes


device = "cpu"


compiled_f = torch.compile(
    f,
    backend=CuteInductor(
        replacement_configs=[
            ReplacementConfig(
                search_function=search, replacement_function=replace, example_inputs=torch.randn(8, 8, device=device)
            )
        ]
    ).compiler,
)(torch.randn(8, 8, device=device))

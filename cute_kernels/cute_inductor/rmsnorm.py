from inspect import getfullargspec

import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..kernels import rmsnorm_cute
from .constants import CALL_FUNCTION
from .utils import parse_args_and_kwargs_to_kwargs


def replace_rmsnorm(gm: GraphModule, node: Node) -> None:
    if not (node.op == CALL_FUNCTION and node.target == torch.rms_norm):
        return

    signature = getfullargspec(F.rms_norm).args
    kwargs = parse_args_and_kwargs_to_kwargs(signature=signature, args=node.args, kwargs=node.kwargs)

    kwargs["x"] = kwargs.pop("input")

    x = kwargs["x"]
    normalized_shape = kwargs.pop("normalized_shape")

    if normalized_shape == (x.size(-1),):
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(rmsnorm_cute, kwargs=kwargs)

        print("replacing with rmsnorm_cute")

        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

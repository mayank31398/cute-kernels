import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..kernels import swiglu_unchunked_cute
from .constants import CALL_METHOD


def replace_swiglu_unchunked(gm: GraphModule, node: Node) -> None:
    if not (node.op == CALL_METHOD and node.target == torch.chunk.__name__):
        return

    x_dim = node.kwargs.get("input", node.args[0]).dim()

    chunks = node.kwargs.get("chunks", node.args[1])
    if chunks != 2:
        return

    dim = node.kwargs.get("dim", node.args[2])
    # dim should be last dim or skip
    if dim not in [-1, x_dim - 1]:
        return

    print(node)

    # if len(node.args) == 2 and node.args[1] == 2:
    #     with gm.graph.inserting_after(node):
    #         # Create a new node for the custom chunk_silu function
    #         new_node = gm.graph.call_function(chunk_silu, args=(node.args[0],))

    #     # Replace all uses of the old node with the new node
    #     node.replace_all_uses_with(new_node)
    #     gm.graph.erase_node(node)

    # node.replace_all_uses_with(new_node)
    # gm.graph.erase_node(node)

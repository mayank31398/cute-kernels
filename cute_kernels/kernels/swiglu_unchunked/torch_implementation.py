import torch
import torch.nn.functional as F


def swiglu_unchunked_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return x[0] * F.silu(x[1])

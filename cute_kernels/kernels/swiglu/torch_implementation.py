import torch
import torch.nn.functional as F


def swiglu_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return up * F.silu(gate)


def swiglu_packed_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return x[0] * F.silu(x[1])

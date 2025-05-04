import torch
import torch.nn.functional as F


def swiglu_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return up * F.silu(gate)


def swiglu_packed_torch(x: torch.Tensor) -> torch.Tensor:
    up, gate = x.chunk(2, dim=-1)
    return swiglu_torch(gate=gate, up=up)

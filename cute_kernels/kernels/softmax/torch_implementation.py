import torch
import torch.nn.functional as F


def softmax_torch(x: torch.Tensor, logits_multiplier: float = 1) -> torch.Tensor:
    return F.softmax(x * logits_multiplier, dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...math import divide_if_divisible


def rnn_torch(input: torch.Tensor, weight: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
    B, S, N, H = input.size()
    output = torch.empty_like(input)

    if input_state is None:
        input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

    weight = weight.unsqueeze(0)
    input = input.unsqueeze(-2)

    # input -> (B, S, N, 1, H)
    # weight -> (1, N, H, H)
    # input_state -> (B, N, H)

    for s in range(S):
        input_state = input_state.unsqueeze(-2)

        # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
        input_state = input_state @ weight + input[:, s, ...]

        input_state = input_state.float()
        input_state = F.tanh(input_state)
        input_state = input_state.type_as(input)

        input_state = input_state.squeeze(-2)

        output[:, s, ...] = input_state

    return output


class RNNTorch(nn.Module):
    def __init__(
        self, input_size: int, state_size: int, output_size: int, num_heads: int, add_bias: bool = True
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads

        self.input_head_dim = divide_if_divisible(self.input_size, self.num_heads)
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads)

        self.input_projection = nn.Linear(self.input_size, self.state_size, bias=add_bias)
        self.output_projection = nn.Linear(self.state_size, self.output_size, bias=False)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

        self.reset_parameters()

    def forward(self, input: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length, _ = input.size()

        input = self.input_projection(input)
        input = input.view(batch_size, sequence_length, self.num_heads, -1)

        input = self._rnn(input, input_state)

        input = input.view(batch_size, sequence_length, -1)
        input = self.output_projection(input)

        return input

    def _rnn(self, input: torch.Tensor, input_state: torch.Tensor | None) -> torch.Tensor:
        return rnn_torch(input=input, weight=self.state_weight, input_state=input_state)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)

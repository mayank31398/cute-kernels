import torch


def set_inductor_defaults() -> None:
    torch._dynamo.config.cache_size_limit = 64

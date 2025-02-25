import torch

from cute_kernels import (
    CuteInductor,
    rmsnorm_replacement_config,
    rmsnorm_torch,
    swiglu_unchunked_replacement_config,
    swiglu_unchunked_torch,
)


def f(x):
    x = x * 4
    x = x + 3
    x = swiglu_unchunked_torch(x)
    x = x - 3
    x = rmsnorm_torch(x, torch.randn(x.size(-1), device=x.device), eps=None)
    return x


device = torch.cuda.current_device()


compiled_f = torch.compile(
    f,
    backend=CuteInductor(
        replacement_configs=[swiglu_unchunked_replacement_config, rmsnorm_replacement_config]
    ).compiler,
)(torch.randn(8, 8, device=device))

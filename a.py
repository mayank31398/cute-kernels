import torch
import torch._inductor.config as config
import torch.nn.functional as F
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, joint_fwd_bwd, register_replacement

from cute_kernels import CuteInductor, swiglu_unchunked_cute, swiglu_unchunked_torch


def f(x):
    return swiglu_unchunked_cute(x)


class _CustomPass(PatternMatcherPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        self.apply(g)
        print(g)


with config.patch(joint_custom_post_pass=_CustomPass()):
    my_args = [
        torch.empty([10, 10], device=torch.cuda.current_device(), requires_grad=True),
    ]

    invoked = False

    def extra_check(match):
        global invoked
        invoked = True
        return True

    register_replacement(
        swiglu_unchunked_torch,
        f,
        my_args,
        joint_fwd_bwd,
        [config.joint_custom_post_pass],
        extra_check=extra_check,
    )

    compiled = torch.compile(swiglu_unchunked_torch, dynamic=True, backend=CuteInductor().compiler)

    x = torch.randn([8, 8], device=torch.cuda.current_device())
    x = x.detach().requires_grad_()

    x_clone = x.clone().detach().requires_grad_()

    z = compiled(x)
    z_clone = swiglu_unchunked_torch(x_clone)

    print(z - z_clone)

    z.sum().backward()
    z_clone.sum().backward()

    print(x.grad - x_clone.grad)

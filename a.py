import torch
import torch._inductor.config as config
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, register_replacement

from cute_kernels import CuteInductor, swiglu_unchunked_cute, swiglu_unchunked_torch


def f(x):
    return swiglu_unchunked_cute(x)


saved_graph = None


class _CustomPass(PatternMatcherPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        self.apply(g)
        global saved_graph
        saved_graph = g


with config.patch(
    # leave custom pass only in post_grad_passes()
    pattern_matcher=False,
    # define pattern match as custom post grad opt pass
    post_grad_custom_pre_pass=None,
    post_grad_custom_post_pass=_CustomPass(),
):
    my_args = [torch.empty([10, 10], device="cpu")]

    invoked = False

    def extra_check(match):
        global invoked
        invoked = True
        return True

    register_replacement(
        # add,
        # sym_minus,
        swiglu_unchunked_torch,
        f,
        my_args,
        fwd_only,
        [config.post_grad_custom_post_pass],
        extra_check=extra_check,
    )

    compiled = torch.compile(swiglu_unchunked_torch, dynamic=True, backend=CuteInductor().compiler)

    x = torch.rand([8, 8], device=torch.cuda.current_device())
    z = compiled(x)

    print(z - swiglu_unchunked_torch(x))
    print(saved_graph)

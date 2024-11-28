from .add import add_scalar_cute, add_scalar_torch, add_tensor_cute, add_tensor_torch
from .contiguous_count import contiguous_count_cute
from .embedding import embedding_cute, embedding_torch
from .inductor import set_inductor_defaults
from .rmsnorm import rmsnorm_cute, rmsnorm_torch
from .scattermoe import MoE_Torch, MoE_Triton
from .swiglu import swiglu_cute, swiglu_torch


set_inductor_defaults()

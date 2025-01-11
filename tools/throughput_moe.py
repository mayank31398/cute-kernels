from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, LightningMoE, MoE_Torch, ScatterMoE, device_synchronize, swiglu_unchunked_cute


n = 100

headers = ["dtype", "torch", "cuda", "triton"]
kernels = [
    MoE_Torch(64, 2, 4096, 512, swiglu_unchunked_cute, True, False, 0.02),
    ScatterMoE(64, 2, 4096, 512, swiglu_unchunked_cute, True, False, 0.02),
    LightningMoE(64, 2, 4096, 512, swiglu_unchunked_cute, True, False, 0.02),
]

for kernel in kernels[1:]:
    kernel.load_state_dict(kernels[0].state_dict())

for kernel in kernels:
    kernel.to(torch.cuda.current_device())

table = []

with torch.no_grad():
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        row = [str(dtype)]
        for kernel in kernels:
            x = torch.randn(4, 4096, 4096, device=torch.cuda.current_device(), dtype=dtype)

            for i in range(n):
                z = kernel(x)

            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            s.record()
            for i in range(n):
                z = kernel(x)
            e.record()

            device_synchronize()

            row.append(s.elapsed_time(e) / n)
        table.append(row)


print(tabulate(table, headers=headers))

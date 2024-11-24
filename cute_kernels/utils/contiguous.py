from typing import Any

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


def make_contiguous(x: Any) -> Any:
    return x.contiguous() if isinstance(x, torch.Tensor) else x


def ensure_same_strides(*args, force_contiguous: bool = False) -> list[torch.Tensor]:
    if force_contiguous:
        output = [make_contiguous(arg) for arg in args]
    else:
        mismatch = False
        expected_stride = None

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if expected_stride is None:
                    expected_stride = arg.stride()
                elif arg.stride() != expected_stride:
                    mismatch = True
                    break

        output = [make_contiguous(arg) for arg in args] if mismatch else args

    return output

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ReplacementConfig:
    search_function: Callable
    replacement_function: Callable
    example_inputs_function: Callable
    prepare_inputs_function: Callable

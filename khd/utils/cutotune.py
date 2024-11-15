import inspect
import os
from collections import defaultdict
from itertools import product
from time import perf_counter
from typing import Any, Callable, get_args

import torch
import torch.distributed

from .synchronization import device_synchronize


_DEBUG_CUTOTUNE = bool(os.getenv("DEBUG_CUTOTUNE", 0))
_DISABLE_CUTOTUNE = bool(os.getenv("DISABLE_CUTOTUNE", 0))
_SEPARATOR = "."


class CutoTuneConfig:
    def __init__(self, config: dict, condition: Callable = None) -> None:
        self.config = config
        self.condition = condition

    def get_key_values(self) -> dict:
        return self.config

    def is_condition_valid(self, **kwargs) -> bool:
        return True if self.condition is None else self.condition(**kwargs)

    def __repr__(self) -> str:
        return str(self.config)


class CutoTuneParameter: ...


class _CutoTune:
    def __init__(
        self,
        function: Callable,
        configs: list[CutoTuneConfig],
        triggers: set[str] = set(),
        warmup_iterations: int = 5,
        benchmark_iterations: int = 100,
        in_place_op: bool = False,
    ) -> None:
        self.function = function
        self.configs = configs
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.in_place_op = in_place_op

        self.signature = inspect.getfullargspec(function)

        self._setup_overrideables()
        self._setup_trigger_map(triggers)
        self._check_configs()

        if self.in_place_op:
            raise NotImplementedError()

        self.best_configs = {}

    def __call__(self, *args, **kwargs) -> Any:
        if _DISABLE_CUTOTUNE:
            self._check_no_args_are_cutotune_overrideable(*args, **kwargs)
            output = self.function(*args, **kwargs)
        else:
            override_cutotune_parameters = self._check_all_or_no_args_are_cutotune_overrideable(*args, **kwargs)

            lookup_key = self._get_lookup_key(*args, **kwargs)

            if lookup_key not in self.best_configs:
                best_config, best_time = self._cutotune(*args, **kwargs)

                if _DEBUG_CUTOTUNE and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                    print(f"config {best_config} achieved the best time ({best_time} sec) for {lookup_key}")

                self.best_configs[lookup_key] = best_config

            output = self.function(
                **self._get_function_arguments(
                    config=self.best_configs[lookup_key],
                    args=args,
                    kwargs=kwargs,
                    override_allowed=override_cutotune_parameters,
                )
            )

        return output

    def _check_no_args_are_cutotune_overrideable(self, *args, **kwargs) -> None:
        for i, value in enumerate(args):
            assert not isinstance(
                value, CutoTuneParameter
            ), f"{self.signature.args[i]} should not be CutoTuneParameter"

        for variable_name, value in kwargs.items():
            assert not isinstance(value, CutoTuneParameter), f"{variable_name} should not be CutoTuneParameter"

    def _check_all_or_no_args_are_cutotune_overrideable(self, *args, **kwargs) -> bool:
        num_cutotune_overrideables = 0

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]

            if isinstance(value, CutoTuneParameter):
                assert variable_name in self.overrideables
                num_cutotune_overrideables += 1

        for variable_name, value in kwargs.items():
            if isinstance(value, CutoTuneParameter):
                assert variable_name in self.overrideables
                num_cutotune_overrideables += 1

        assert num_cutotune_overrideables in [
            0,
            len(self.overrideables),
        ], f"invalid number of CutoTuneParameter arguments, should be either 0 or {len(self.overrideables)}"

        return num_cutotune_overrideables == 0

    def _get_function_arguments(
        self, config: CutoTuneConfig, args: list, kwargs: dict, override_allowed: bool
    ) -> dict:
        # copy the best_config first so we can override with args or kwargs
        result = {variable_name: value for variable_name, value in config.get_key_values().items()}

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]

            if override_allowed or variable_name not in result:
                result[variable_name] = value

        for variable_name, value in kwargs.items():
            if override_allowed or variable_name not in result:
                result[variable_name] = value

        return result

    @torch.inference_mode()
    def _cutotune(self, *args, **kwargs) -> tuple[CutoTuneConfig, float]:
        best_config = None
        best_time = float("inf")

        for config in self.configs:
            if not config.is_condition_valid(
                **self._get_function_arguments(
                    config=CutoTuneConfig({}), args=args, kwargs=kwargs, override_allowed=False
                )
            ):
                continue

            elapsed_time = self._run_benchmark(
                **self._get_function_arguments(config=config, args=args, kwargs=kwargs, override_allowed=False),
            )

            if elapsed_time < best_time:
                best_config = config
                best_time = elapsed_time

        assert best_config is not None, "no best_config found, check that at least 1 cutotune config is valid"

        return best_config, best_time

    def _get_lookup_key(self, *args, **kwargs) -> Any:
        lookup_key = []

        def _maybe_add_key(variable_name: str, value) -> None:
            if variable_name not in self.variable_name_trigger_map:
                return

            triggers = self.variable_name_trigger_map[variable_name]

            if isinstance(value, torch.Tensor):
                for trigger in triggers:
                    if trigger is None:
                        assert len(triggers) == 1
                        trigger = lambda tensor: (tensor.dtype, tensor.size(), tensor.stride())

                    lookup_key.append(f"{variable_name} = {trigger(value)}")
            else:
                assert len(triggers) == 1
                assert (
                    triggers[0] is None
                ), f"trigger ({variable_name}) is not a tensor and shouldn't have a functional trigger"

                lookup_key.append(f"{variable_name} = {value}")

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]
            _maybe_add_key(variable_name, value)

        for variable_name, value in kwargs.items():
            _maybe_add_key(variable_name, value)

        return tuple(lookup_key)

    def _run_benchmark(self, **kwargs: dict) -> float:
        device_synchronize()

        for _ in range(self.warmup_iterations):
            self.function(**kwargs)

        device_synchronize()
        start_time = perf_counter()

        for _ in range(self.benchmark_iterations):
            self.function(**kwargs)

        device_synchronize()
        end_time = perf_counter()
        elapsed_time = end_time - start_time

        return elapsed_time / self.benchmark_iterations

    def _check_configs(self) -> None:
        for config in self.configs:
            assert (
                set(config.get_key_values().keys()) == self.overrideables
            ), "cutotune configs don't match the expected function signature"

    def _setup_overrideables(self) -> None:
        self.overrideables = set()
        for key in self.signature.annotations:
            if CutoTuneParameter in get_args(self.signature.annotations[key]):
                self.overrideables.add(key)

    def _setup_trigger_map(self, triggers: set[str]) -> None:
        assert isinstance(triggers, set), "triggers should be a set"

        self.variable_name_trigger_map = defaultdict(list)

        for trigger in triggers:
            variable_name, trigger = self._parse_trigger(trigger)
            self.variable_name_trigger_map[variable_name].append(trigger)

        # filter to remove all triggers if None, this is usefull for Tensor based triggers
        for variable_name in self.variable_name_trigger_map:
            if None in self.variable_name_trigger_map[variable_name]:
                self.variable_name_trigger_map[variable_name] = [None]

            assert (
                variable_name in self.signature.args
            ), f"unexpected variable_name ({variable_name}) found in triggers"

        for variable_name in self.overrideables:
            assert (
                variable_name not in self.variable_name_trigger_map
            ), "trigger can't be an instance of CutoTuneParameter"

    def _parse_trigger(self, trigger: str) -> tuple[str, Callable]:
        split_trigger = trigger.split(_SEPARATOR)
        variable_name = split_trigger[0]

        if len(split_trigger) == 1:
            func = None
        elif len(split_trigger) == 2:
            if split_trigger[1] == "dtype":
                func = lambda tensor: tensor.dtype
            elif split_trigger[1] in ["size()", "shape"]:
                func = lambda tensor: tensor.size()
            elif split_trigger[1] == "stride()":
                func = lambda tensor: tensor.stride()
            elif split_trigger[1].startswith("size"):
                dim = int(func[5:][:-1])
                func = lambda tensor: tensor.size(dim)
            elif split_trigger[1].startswith("shape"):
                dim = int(func[6:][:-1])
                func = lambda tensor: tensor.size(dim)
            elif split_trigger[1].startswith("stride"):
                dim = int(func[7:][:-1])
                func = lambda tensor: tensor.stride(dim)
            else:
                raise ValueError(f"unexpected triggeer found ({trigger})")

        return variable_name, func

    def __enter__(self) -> Any:
        return

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Any:
        return


def cutotune(
    configs: list[CutoTuneConfig],
    triggers: set[str] = set(),
    warmup_iterations: int = 5,
    benchmark_iterations: int = 100,
    in_place_op: bool = False,
) -> _CutoTune:
    def inner(function: Callable) -> Callable:
        return _CutoTune(
            function=function,
            configs=configs,
            triggers=triggers,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            in_place_op=in_place_op,
        ).__call__

    return inner


def get_cartesian_product_cutotune_configs(
    condition: Callable = None, **kwargs: dict[str, list]
) -> list[CutoTuneConfig]:
    configs = []
    all_values = product(*list(kwargs.values()))

    for values in all_values:
        config = {key: value for key, value in zip(kwargs.keys(), values)}
        config = CutoTuneConfig(config, condition=condition)
        configs.append(config)

    return configs

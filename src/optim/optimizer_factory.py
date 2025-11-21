from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, Any

from torch import nn, optim

from ..config import OptimizationConfig


@dataclass(frozen=True)
class OptimizerPreset:
    name: str
    optimizer_type: str
    params: Dict[str, float | bool | tuple]


# Optimizer presets have to be created
OPTIMIZER_PRESETS: Dict[str, OptimizerPreset] = {
    "sgd_baseline": OptimizerPreset(
        name="sgd_baseline",
        optimizer_type="SGD",
        params={"lr": 0.1, "momentum": 0.0, "nesterov": False, "weight_decay": 0.0},
    ),
    "adam_default": OptimizerPreset(
        name="adam_default",
        optimizer_type="Adam",
        params={"lr": 3e-4, "weight_decay": 5e-4},
    ),
}


def available_presets() -> Iterable[str]:
    return OPTIMIZER_PRESETS.keys()


def build_optimizer(
    model: nn.Module,
    config: OptimizationConfig,
) -> optim.Optimizer:
    preset = OPTIMIZER_PRESETS.get(config.profile)
    if preset is None:
        raise KeyError(
            f"Unknown optimizer profile '{config.profile}'. Available: {list(available_presets())}"
        )

    # 1. Get the optimizer class dynamically
    if not hasattr(optim, preset.optimizer_type):
        raise ValueError(f"Unsupported optimizer type '{preset.optimizer_type}'.")

    opt_cls = getattr(optim, preset.optimizer_type)

    # 2. Merge config overrides
    params: Dict[str, Any] = dict(preset.params)
    overrides = {
        key: value
        for key, value in vars(config).items()
        if key not in {"profile"} and value is not None
    }
    params.update(overrides)

    # 3. Filter params to only those accepted by the optimizer class
    valid_args = set(inspect.signature(opt_cls).parameters.keys())
    filtered_kwargs = {k: v for k, v in params.items() if k in valid_args}

    return opt_cls(model.parameters(), **filtered_kwargs)

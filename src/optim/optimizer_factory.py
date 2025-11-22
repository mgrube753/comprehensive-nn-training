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


OPTIMIZER_PRESETS: Dict[str, OptimizerPreset] = {
    "sgd_0": OptimizerPreset(
        name="sgd_0",
        optimizer_type="SGD",
        params={"lr": 0.1, "momentum": 0.0, "nesterov": False, "weight_decay": 0.0},
    ),
    "sgd_1": OptimizerPreset(
        name="sgd_1",
        optimizer_type="SGD",
        params={"lr": 0.01, "momentum": 0.9, "nesterov": True, "weight_decay": 1e-4},
    ),
    # rescnn 0.65 test_acc / 1.24 test_loss
    "sgd_2": OptimizerPreset(
        name="sgd_2",
        optimizer_type="SGD",
        params={"lr": 0.05, "momentum": 0.95, "nesterov": True, "weight_decay": 5e-4},
    ),
    # rescnn 0.65 test_acc / 1.30 test_loss
    "adam_0": OptimizerPreset(
        name="adam_0",
        optimizer_type="Adam",
        params={},
    ),
    # rescnn 0.66 test_acc / 1.24 test_loss
    "adam_1": OptimizerPreset(
        name="adam_1",
        optimizer_type="AdamW",
        params={"lr": 0.001, "weight_decay": 1e-4},
    ),
    "adam_2": OptimizerPreset(
        name="adam_2",
        optimizer_type="AdamW",
        params={"lr": 0.0005, "weight_decay": 5e-4},
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

    if not hasattr(optim, preset.optimizer_type):
        raise ValueError(f"Unsupported optimizer type '{preset.optimizer_type}'.")

    opt_cls = getattr(optim, preset.optimizer_type)

    params: Dict[str, Any] = dict(preset.params)
    overrides = {
        key: value
        for key, value in vars(config).items()
        if key not in {"profile"} and value is not None
    }
    params.update(overrides)

    valid_args = set(inspect.signature(opt_cls).parameters.keys())
    filtered_kwargs = {k: v for k, v in params.items() if k in valid_args}

    return opt_cls(model.parameters(), **filtered_kwargs)

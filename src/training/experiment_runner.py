from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from typing import Iterable, List

from ..config import ExperimentConfig
from ..data.datamodule import CIFAR100DataModule
from ..models.sample_cnn import SampleCNN
from .trainer import Trainer

DEFAULT_PROFILES = [
    "sgd_baseline",
    "adam_default",
]


def run_experiments(
    profiles: Iterable[str], base_config: ExperimentConfig | None = None
) -> None:
    config_template = base_config or ExperimentConfig()
    for profile in profiles:
        config = deepcopy(config_template)
        config.optimization.profile = profile
        config.mlflow.run_name = (
            f"{profile}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
        model = SampleCNN()
        data = CIFAR100DataModule(config.dataset)
        trainer = Trainer(model, data, config)
        result = trainer.fit()
        val_loss_str = (
            f"{result.best_val_metrics.loss:.4f}"
            if result.best_val_metrics is not None
            else "n/a"
        )
        print(
            f"Profile {profile}: epochs={result.epochs_ran}, "
            f"val_loss={val_loss_str}, "
            f"test_acc={result.test_metrics.accuracy:.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CIFAR100 experiments")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=DEFAULT_PROFILES,
        help="Optimizer profiles to run sequentially",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override train batch size",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = ExperimentConfig()
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
    run_experiments(args.profiles, config)


if __name__ == "__main__":
    main()

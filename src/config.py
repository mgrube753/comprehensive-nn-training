from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, Tuple

from .models.cnns import CNNConfig


@dataclass
class DatasetConfig:
    data_root: str = "./data"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 128
    test_batch_size: Optional[int] = None
    num_workers: int = 8
    pin_memory: bool = True
    download: bool = True
    seed: int = 42


@dataclass
class OptimizationConfig:
    profile: str = "sgd_momentum"
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    momentum: Optional[float] = None
    nesterov: Optional[bool] = None
    betas: Optional[Tuple[float, float]] = None


@dataclass
class SchedulerConfig:
    enabled: bool = True
    factor: float = 0.5
    min_lr: float = 5e-5
    patience_epochs: int = 2
    cooldown_epochs: int = 0
    max_reductions: int = 3


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience_epochs: int = 4
    min_delta: float = 1e-4


@dataclass
class CheckpointConfig:
    output_dir: str = "./checkpoints"
    save_best_only: bool = True
    keep_last: bool = True


@dataclass
class TrainingLoopConfig:
    epochs: int = 20
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = None
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1


@dataclass
class MLflowConfig:
    tracking_uri: Optional[str] = None
    experiment_name: str = field(
        default_factory=lambda: f"cifar100-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: CNNConfig = field(default_factory=CNNConfig)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

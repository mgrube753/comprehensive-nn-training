from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import mlflow
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ExperimentConfig
from ..data.datamodule import CIFAR100DataModule
from ..optim.optimizer_factory import build_optimizer
from .checkpoint import CheckpointManager
from .early_stopping import EarlyStoppingGuard
from .lr_scheduler import AdaptiveLRScheduler
from .metrics import EpochMetrics, MetricState, finalize_metrics


@dataclass
class TrainingResult:
    epochs_ran: int
    best_val_metrics: Optional[EpochMetrics]
    test_metrics: EpochMetrics
    checkpoint_path: Optional[Path]
    run_id: Optional[str]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data_module: CIFAR100DataModule,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = self.resolve_device(config.training.device)

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(model, config.optimization)

        self.scheduler = (
            AdaptiveLRScheduler(self.optimizer, config.scheduler)
            if config.scheduler.enabled
            else None
        )
        self.early_stopper = (
            EarlyStoppingGuard(config.early_stopping)
            if config.early_stopping.enabled
            else None
        )
        self.checkpointer = CheckpointManager(config.checkpoint)

        self.use_amp = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.seed_everything(config.dataset.seed)
        self.global_step = 0

    def fit(self) -> TrainingResult:
        self.data_module.prepare_data()
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()

        best_val_metrics: Optional[EpochMetrics] = None
        checkpoint_path: Optional[Path] = None
        epochs_ran = 0

        with self.mlflow_run() as run_id:
            self.log_params_to_mlflow()
            initial_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", initial_lr, step=0)

            for epoch in range(1, self.config.training.epochs + 1):
                epochs_ran = epoch
                train_metrics = self.run_epoch(train_loader, training=True, epoch=epoch)
                self.log_metrics(train_metrics, split="train", step=epoch)

                if epoch % self.config.training.val_every_n_epochs == 0:
                    val_metrics = self.run_epoch(
                        val_loader, training=False, epoch=epoch
                    )
                    if (
                        best_val_metrics is None
                        or val_metrics.loss < best_val_metrics.loss
                    ):
                        best_val_metrics = val_metrics
                    self.log_metrics(val_metrics, split="val", step=epoch)

                    checkpoint_path = self.checkpointer.maybe_save(
                        model_state=self.model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        epoch=epoch,
                        val_loss=val_metrics.loss,
                    )
                    if checkpoint_path is not None:
                        self.log_artifact(checkpoint_path)

                    if self.scheduler is not None:
                        lr_change = self.scheduler.step(val_metrics.loss)
                        if lr_change is not None:
                            old_lr, new_lr = lr_change
                            print(f"\n{'='*60}")
                            print(f"Learning Rate Reduced: {old_lr:.6f} → {new_lr:.6f}")
                            print(f"{'='*60}\n")
                            mlflow.log_metric("lr", new_lr, step=epoch)

                    if self.early_stopper is not None:
                        if self.early_stopper.update(val_metrics.loss):
                            print(f"\n{'='*60}")
                            print(f"Early Stopping Triggered at Epoch {epoch}")
                            print(
                                f"Best validation loss: {self.early_stopper.best_loss:.4f}"
                            )
                            print(
                                f"No improvement for {self.config.early_stopping.patience_epochs} epochs"
                            )
                            print(f"{'='*60}\n")
                            break

            best_checkpoint = self.checkpointer.best_checkpoint()
            if best_checkpoint is not None:
                state = torch.load(best_checkpoint, map_location=self.device)
                self.model.load_state_dict(state["model_state_dict"])

            test_metrics = self.run_epoch(test_loader, training=False, epoch=epochs_ran)
            mlflow.log_metric("test_loss", test_metrics.loss)
            mlflow.log_metric("test_accuracy", test_metrics.accuracy)

        return TrainingResult(
            epochs_ran=epochs_ran,
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            checkpoint_path=self.checkpointer.best_checkpoint(),
            run_id=run_id,
        )

    def run_epoch(
        self, dataloader: DataLoader, *, training: bool, epoch: int
    ) -> EpochMetrics:
        state = MetricState()
        self.model.train(mode=training)

        phase = "Train" if training else "Val"
        total_epochs = self.config.training.epochs
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{total_epochs} [{phase}]",
            unit="batch",
            leave=True,
            ncols=140,
        )

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            for batch_idx, (inputs, targets) in enumerate(pbar, start=1):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    self.global_step += 1

                state.update(loss, logits, targets)

                current = state.current_metrics()
                pbar.set_postfix(
                    loss=f"{current['loss']:.4f}",
                    acc=f"{current['acc']:.4f}",
                )

                if (
                    training
                    and self.global_step % self.config.training.log_every_n_steps == 0
                ):
                    mlflow.log_metrics(current, step=self.global_step)

        return finalize_metrics(state)

    def resolve_device(self, requested: str) -> torch.device:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def log_params_to_mlflow(self) -> None:
        params = self.flatten_dict(self.config.to_dict())
        mlflow.log_params(params)

    def log_metrics(self, metrics: EpochMetrics, *, split: str, step: int) -> None:
        mlflow.log_metrics(metrics.as_dict(split), step=step)

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def flatten_dict(
        self, values: Dict[str, object], prefix: Optional[str] = None
    ) -> Dict[str, object]:
        items: Dict[str, object] = {}
        for key, value in values.items():
            compound_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(self.flatten_dict(value, compound_key))
            else:
                items[compound_key] = value
        return items

    @contextmanager
    def mlflow_run(self) -> Iterable[Optional[str]]:
        tracking = self.config.mlflow
        if tracking.tracking_uri:
            mlflow.set_tracking_uri(tracking.tracking_uri)
        mlflow.set_experiment(tracking.experiment_name)
        run_name = tracking.run_name or self.default_run_name()
        with mlflow.start_run(run_name=run_name, tags=tracking.tags) as run:
            yield run.info.run_id

    def default_run_name(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{self.config.optimization.profile}-{timestamp}"

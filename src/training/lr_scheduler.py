from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..config import SchedulerConfig


class AdaptiveLRScheduler:
    def __init__(self, optimizer: Optimizer, config: SchedulerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self._reductions = 0

        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience_epochs,
            cooldown=config.cooldown_epochs,
            min_lr=config.min_lr,
        )

    def step(self, val_loss: float) -> Optional[float]:
        if not self.config.enabled:
            return None

        if self._reductions >= self.config.max_reductions:
            return None

        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(val_loss)
        new_lr = self.optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            self._reductions += 1
            return new_lr

        return None

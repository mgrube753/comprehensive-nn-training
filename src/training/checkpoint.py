from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..config import CheckpointConfig


class CheckpointManager:
    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self.best_loss = float("inf")
        self.best_path: Optional[Path] = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(
        self, *, model_state: dict, optimizer_state: dict, epoch: int, val_loss: float
    ) -> Optional[Path]:
        if self.config.save_best_only and val_loss >= self.best_loss:
            return None

        path = self.output_dir / f"checkpoint-epoch{epoch:03d}.pt"
        torch.save(
            {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            path,
        )

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_path = path

        return path

    def best_checkpoint(self) -> Optional[Path]:
        return self.best_path

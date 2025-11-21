from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class MetricState:
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        batch_size = targets.size(0)
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

        predictions = logits.argmax(dim=1)
        self.total_correct += (predictions == targets).sum().item()

    def current_metrics(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {"loss": 0.0, "acc": 0.0}

        avg_loss = self.total_loss / self.total_samples
        accuracy = self.total_correct / self.total_samples

        return {
            "loss": avg_loss,
            "acc": accuracy,
        }


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        key_prefix = f"{prefix}_" if prefix else ""
        return {
            f"{key_prefix}loss": self.loss,
            f"{key_prefix}accuracy": self.accuracy,
        }


def finalize_metrics(state: MetricState) -> EpochMetrics:
    if state.total_samples == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0)

    avg_loss = state.total_loss / state.total_samples
    accuracy = state.total_correct / state.total_samples

    return EpochMetrics(loss=avg_loss, accuracy=accuracy)

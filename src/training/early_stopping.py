from __future__ import annotations

from dataclasses import dataclass

from ..config import EarlyStoppingConfig


@dataclass
class EarlyStoppingState:
    patience_counter: int = 0
    best_loss: float = float("inf")
    should_stop: bool = False


class EarlyStoppingGuard:
    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.state = EarlyStoppingState()

    def update(self, val_loss: float) -> bool:
        state = self.state
        if val_loss < state.best_loss - self.config.min_delta:
            state.best_loss = val_loss
            state.patience_counter = 0
            return False

        state.patience_counter += 1
        if (
            self.config.enabled
            and state.patience_counter >= self.config.patience_epochs
        ):
            state.should_stop = True
        return state.should_stop

    @property
    def best_loss(self) -> float:
        return self.state.best_loss

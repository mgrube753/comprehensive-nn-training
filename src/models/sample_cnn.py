from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    init,
)


@dataclass
class SampleCNNConfig:
    in_channels: int = 3
    num_classes: int = 100
    width: int = 64
    dropout: float = 0.2


class SampleCNN(Module):
    def __init__(self, config: SampleCNNConfig | None = None) -> None:
        super().__init__()
        c = config or SampleCNNConfig()

        def conv_block(ic, oc):
            return Sequential(
                Conv2d(ic, oc, 3, 1, 1, bias=False),
                BatchNorm2d(oc),
                ReLU(True),
                Conv2d(oc, oc, 3, 1, 1, bias=False),
                BatchNorm2d(oc),
                ReLU(True),
                Dropout(c.dropout),
                MaxPool2d(2),
            )

        self.features = Sequential(
            conv_block(c.in_channels, c.width),
            conv_block(c.width, c.width * 2),
            conv_block(c.width * 2, c.width * 4),
        )
        self.classifier = Sequential(
            Flatten(),
            Linear(c.width * 4 * 4 * 4, c.width * 4),
            ReLU(True),
            Dropout(c.dropout),
            Linear(c.width * 4, c.num_classes),
        )
        self.apply(self.init_weights)

    def init_weights(self, m: Module) -> None:
        if isinstance(m, Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, Linear):
            init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

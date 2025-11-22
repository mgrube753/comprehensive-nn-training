from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    GELU,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    init,
)


@dataclass
class CNNConfig:
    model_name: str = "sample_cnn"
    in_channels: int = 3
    num_classes: int = 100
    width: int = 64
    dropout: float = 0.2


def create_model(config: CNNConfig) -> Module:
    model_map = {
        "sample_cnn": SampleCNN,
        "res_cnn": ResCNN,
    }

    name = config.model_name.lower()
    if name not in model_map:
        raise ValueError(
            f"Model '{name}' not found. Available: {list(model_map.keys())}"
        )

    return model_map[name](config)


class SampleCNN(Module):
    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        c = config or CNNConfig()

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


class ResCNN(Module):
    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        c = config or CNNConfig()

        class ResBlock(Module):
            def __init__(self, ic, oc, stride=1):
                super().__init__()
                self.conv1 = Conv2d(ic, oc, 3, stride, 1, bias=False)
                self.bn1 = BatchNorm2d(oc)
                self.conv2 = Conv2d(oc, oc, 3, 1, 1, bias=False)
                self.bn2 = BatchNorm2d(oc)
                self.act = GELU()
                self.downsample = None
                if stride != 1 or ic != oc:
                    self.downsample = Sequential(
                        Conv2d(ic, oc, 1, stride, bias=False), BatchNorm2d(oc)
                    )

            def forward(self, x):
                identity = x
                out = self.act(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                return self.act(out + identity)

        self.stem = Sequential(
            Conv2d(c.in_channels, c.width, 3, 1, 1, bias=False),
            BatchNorm2d(c.width),
            GELU(),
        )

        self.features = Sequential(
            ResBlock(c.width, c.width),
            ResBlock(c.width, c.width * 2, stride=2),
            ResBlock(c.width * 2, c.width * 2),
            ResBlock(c.width * 2, c.width * 4, stride=2),
            ResBlock(c.width * 4, c.width * 4),
        )

        self.head = Sequential(
            AdaptiveAvgPool2d((1, 1)),
            Flatten(),
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
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)

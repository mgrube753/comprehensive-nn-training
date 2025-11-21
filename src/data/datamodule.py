from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Generator
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from ..config import DatasetConfig


@dataclass
class DataSplit:
    train: Dataset
    val: Dataset
    test: Dataset


class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform: Compose | None = None) -> None:
        self.subset, self.transform = subset, transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x, y = self.subset[index]
        return (self.transform(x), y) if self.transform else (x, y)


class CIFAR100DataModule:
    def __init__(self, config: DatasetConfig) -> None:
        self.config, self.splits = config, None
        stats = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        self.tf_train = Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*stats),
            ]
        )
        self.tf_eval = Compose([ToTensor(), Normalize(*stats)])

    def prepare_data(self) -> None:
        for t in [True, False]:
            CIFAR100(self.config.data_root, train=t, download=self.config.download)

    def setup(self) -> None:
        if self.splits:
            return
        full = ConcatDataset(
            [
                CIFAR100(self.config.data_root, train=t, download=False)
                for t in [True, False]
            ]
        )
        lens = [
            math.floor(len(full) * r)
            for r in [self.config.train_ratio, self.config.val_ratio]
        ]
        lens.append(len(full) - sum(lens))
        subs = random_split(
            full, lens, generator=Generator().manual_seed(self.config.seed)
        )
        self.splits = DataSplit(
            TransformedSubset(subs[0], self.tf_train),
            TransformedSubset(subs[1], self.tf_eval),
            TransformedSubset(subs[2], self.tf_eval),
        )

    def loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        bs = (
            self.config.batch_size
            if shuffle
            else (self.config.test_batch_size or self.config.batch_size)
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def ensure(self) -> None:
        self.prepare_data()
        self.setup() if not self.splits else None

    def train_dataloader(self) -> DataLoader:
        self.ensure()
        return self.loader(self.splits.train, True)

    def val_dataloader(self) -> DataLoader:
        self.ensure()
        return self.loader(self.splits.val, False)

    def test_dataloader(self) -> DataLoader:
        self.ensure()
        return self.loader(self.splits.test, False)

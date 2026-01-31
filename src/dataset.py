from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torch import Tensor, stack
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class SimCLRDataset(Dataset):
    def __init__(
        self,
        path: str | Path = "~/data/cifar-10/",
        train: bool = True,
    ):
        transform = T.Compose(
            [
                T.PILToTensor(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

        self.data = CIFAR10(
            root=path,
            train=train,
            transform=transform,
            download=True,
        )

    def __getitem__(
        self,
        idx: int,
    ) -> Tensor:
        original, _ = self.data[idx]

        augment1 = self.sample_transform()(original)
        augment2 = self.sample_transform()(original)

        return stack([augment1, augment2], dim=0)

    def sample_transform(self) -> T.Compose:
        transforms = [
            T.RandomResizedCrop(
                size=32,
                scale=(0.08, 0.5),
                ratio=(0.75, 1.33),
            ),
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.3,
            ),
            T.GaussianBlur(kernel_size=3),
        ]

        return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.data)

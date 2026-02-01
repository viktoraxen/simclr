from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torch import Tensor, nn
from torch.types import Number
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Tensor(CIFAR10):
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

        super().__init__(
            root=path,
            train=train,
            transform=transform,
            download=True,
        )


class SimCLRDataset(CIFAR10Tensor):
    def __getitem__(
        self,
        idx: int,
    ) -> tuple[Tensor, Tensor]:
        original, _ = super().__getitem__(idx)

        augment1 = self.sample_transform()(original)
        augment2 = self.sample_transform()(original)

        return augment1, augment2

    def sample_transform(self) -> T.Compose:
        transforms = [
            T.RandomResizedCrop(
                size=32,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.33),
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomSolarize(threshold=0.5, p=0.2),
            T.GaussianBlur(kernel_size=3),
        ]

        return T.Compose(transforms)


class CIFAR10Extracted(Dataset):
    def __init__(
        self,
        extractor: nn.Module,
        path: str | Path = "~/data/cifar-10/",
        train: bool = True,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        base_dataset = CIFAR10Tensor(path=path, train=train)
        loader = torch.utils.data.DataLoader(
            base_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        extractor = extractor.to(device)
        extractor.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                features = extractor(images)
                embeddings.append(features.cpu())
                labels.append(targets)

        self.embeddings = torch.cat(embeddings, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Number]:
        return self.embeddings[idx], self.labels[idx].item()

    def __len__(self) -> int:
        return len(self.labels)

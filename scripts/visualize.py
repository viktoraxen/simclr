from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torchvision.datasets import CIFAR10

from network import init_model
from visualizer import SimCLRVisualizer


def main():
    models_dir = Path("models")
    model = init_model(models_dir)

    transform = T.Compose(
        [
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = CIFAR10(
        root="~/data/cifar-10/",
        train=False,
        transform=transform,
        download=True,
    )

    visualizer = SimCLRVisualizer(
        model=model,
        dataset=dataset,
        samples=2000,
    )

    visualizer.tsne()


if __name__ == "__main__":
    main()

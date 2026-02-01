from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torchvision.datasets import CIFAR10

from network import ResNet9
from visualizer import SimCLRVisualizer


def select_model(models_dir: Path) -> Path | None:
    models = sorted(models_dir.glob("*.pth")) if models_dir.exists() else []

    if not models:
        return None

    print("Available models:")

    for i, model_path in enumerate(models):
        print(f"  {i + 1}: {model_path.name}")

    choice = int(input("\nSelect model number: "))

    while not (0 < choice <= len(models)):
        choice = int(input("\nInvalid selection, try again: "))

    return models[choice - 1]


def main():
    models_dir = Path("models")
    model_path = select_model(models_dir)

    if model_path is None:
        print("No models found.")
        return

    checkpoint = torch.load(
        model_path,
        weights_only=True,
    )

    model = ResNet9()
    model.load_state_dict(checkpoint["model"])

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
    )

    visualizer = SimCLRVisualizer(
        model=model,
        dataset=dataset,
    )

    visualizer.tsne()


if __name__ == "__main__":
    main()

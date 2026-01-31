"""Visualize SimCLR embeddings using t-SNE.

Shows how well the model clusters images by class, even though
no labels were used during contrastive learning.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2 as T

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network import ResNet6

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def list_models(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.pth"))


def select_model(models: list[Path]) -> Path:
    print("Available models:")
    for i, model_path in enumerate(models):
        print(f"  [{i}] {model_path.name}")

    while True:
        try:
            choice = int(input("\nSelect model number: "))
            if 0 <= choice < len(models):
                return models[choice]
        except ValueError:
            pass
        print("Invalid selection, try again.")


def load_model(model_path: Path, device: str) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = ResNet6()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model


def compute_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    count = 0

    with torch.no_grad():
        for images, targets in dataloader:
            if count >= num_samples:
                break

            images = images.to(device)
            features = model(images)

            embeddings.append(features.cpu())
            labels.append(targets)
            count += len(images)

    embeddings = torch.cat(embeddings)[:num_samples]
    labels = torch.cat(labels)[:num_samples]

    return embeddings, labels


def visualize_tsne(embeddings: torch.Tensor, labels: torch.Tensor) -> None:
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    projected = tsne.fit_transform(embeddings.numpy())

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels.numpy(),
        cmap="tab10",
        alpha=0.6,
        s=10,
    )

    handles, _ = scatter.legend_elements()
    ax.legend(handles, CIFAR10_CLASSES, loc="best", title="Classes")

    ax.set_title("SimCLR Embeddings (t-SNE projection)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(__file__).parent.parent / "models"

    models = list_models(models_dir)
    if not models:
        print(f"No models found in {models_dir}")
        print("Train a model first and save it as models/*.pt")
        sys.exit(1)

    model_path = select_model(models)
    print(f"\nLoading {model_path.name}...")

    model = load_model(model_path, device)

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
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    print("Computing embeddings...")
    embeddings, labels = compute_embeddings(model, dataloader, device)

    visualize_tsne(embeddings, labels)


if __name__ == "__main__":
    main()

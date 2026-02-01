# Add src to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as T
from sklearn.manifold import TSNE
from torchvision.datasets import CIFAR10

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network import ResNet10, load_model

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


def compute_embeddings(model, dataset, samples=2000, device="cuda"):
    """Compute embeddings using the model."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=device == "cuda",
    )

    model.eval()
    model.to(device)

    embeddings = []
    labels = []
    current_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            if current_samples >= samples:
                break

            images = images.to(device)
            features = model(images)
            current_samples += len(images)

            embeddings.append(features.cpu())
            labels.extend(targets.tolist())

    embeddings = torch.cat(embeddings)[:samples]
    labels = torch.tensor(labels[:samples])

    return embeddings, labels


def plot_tsne_subplot(ax, projected, labels, title, highlight_class=None):
    """Plot t-SNE on a 3D axis with optional class highlighting."""
    labels_np = labels.numpy()
    cmap = plt.cm.tab10

    for label in range(10):
        mask = labels_np == label
        if highlight_class is None:
            alpha = 0.6
        elif label == highlight_class:
            alpha = 0.8
        else:
            alpha = 0.05

        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            projected[mask, 2],
            c=[cmap(label / 10)],
            alpha=alpha,
            s=8,
            label=CIFAR10_CLASSES[label],
        )

    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.set_zlabel("t-SNE 3", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(__file__).parent.parent / "models"

    print("Loading models...")
    trained_model = load_model(ResNet10, models_dir / "ResNet10_1.1987.pth", device=device)
    untrained_model = ResNet10()

    transform = T.Compose(
        [
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    print("Loading dataset...")
    dataset = CIFAR10(
        root="~/data/cifar-10/",
        train=False,
        transform=transform,
        download=True,
    )

    print("Computing embeddings for trained model...")
    trained_embeddings, labels = compute_embeddings(trained_model, dataset, device=device)

    print("Computing embeddings for untrained model...")
    untrained_embeddings, _ = compute_embeddings(untrained_model, dataset, device=device)

    print("Running t-SNE for trained model...")
    tsne_trained = TSNE(n_components=3, perplexity=30, random_state=42)
    projected_trained = tsne_trained.fit_transform(trained_embeddings.numpy())

    print("Running t-SNE for untrained model...")
    tsne_untrained = TSNE(n_components=3, perplexity=30, random_state=42)
    projected_untrained = tsne_untrained.fit_transform(untrained_embeddings.numpy())

    # Create 3x2 figure
    fig = plt.figure(figsize=(14, 18))

    automobile_idx = CIFAR10_CLASSES.index("automobile")
    frog_idx = CIFAR10_CLASSES.index("frog")

    # Row 1: All classes
    ax1 = fig.add_subplot(3, 2, 1, projection="3d")
    plot_tsne_subplot(ax1, projected_trained, labels, "Trained (ResNet10_1.1987) - All Classes")

    ax2 = fig.add_subplot(3, 2, 2, projection="3d")
    plot_tsne_subplot(ax2, projected_untrained, labels, "Untrained - All Classes")

    # Row 2: Automobiles highlighted
    ax3 = fig.add_subplot(3, 2, 3, projection="3d")
    plot_tsne_subplot(
        ax3,
        projected_trained,
        labels,
        "Trained - Automobile Highlighted",
        highlight_class=automobile_idx,
    )

    ax4 = fig.add_subplot(3, 2, 4, projection="3d")
    plot_tsne_subplot(
        ax4,
        projected_untrained,
        labels,
        "Untrained - Automobile Highlighted",
        highlight_class=automobile_idx,
    )

    # Row 3: Frogs highlighted
    ax5 = fig.add_subplot(3, 2, 5, projection="3d")
    plot_tsne_subplot(
        ax5, projected_trained, labels, "Trained - Frog Highlighted", highlight_class=frog_idx
    )

    ax6 = fig.add_subplot(3, 2, 6, projection="3d")
    plot_tsne_subplot(
        ax6, projected_untrained, labels, "Untrained - Frog Highlighted", highlight_class=frog_idx
    )

    # Add legend to the figure
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.98),
        markerscale=2,
    )

    plt.suptitle(
        "t-SNE 3D Embeddings Comparison: Trained vs Untrained ResNet10",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path(__file__).parent.parent / "comparison_tsne.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()

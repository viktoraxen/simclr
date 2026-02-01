import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.manifold import TSNE
from torch import Tensor, nn
from torch.utils.data import Dataset

CIFAR10_CLASSES: list[str] = [
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


class SimCLRVisualizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    dataset: Dataset
    embeddings: Tensor | None = None
    labels: Tensor | None = None
    samples: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def tsne(self, perplexity: int = 30):
        embeddings, labels = self._embeddings()

        print("Generating t-SNE plot...")

        tsne = TSNE(n_components=3, perplexity=perplexity)
        projected = tsne.fit_transform(embeddings.numpy())

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            projected[:, 2],
            c=labels.numpy(),
            cmap="tab10",
            alpha=0.6,
            s=10,
        )

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
        ax.set_title("SimCLR Embeddings (3D t-SNE)")

        handles, _ = scatter.legend_elements()
        ax.legend(
            handles,
            CIFAR10_CLASSES,
            loc="best",
        )

        plt.tight_layout()
        plt.show()

    def _embeddings(self) -> tuple[Tensor, Tensor]:
        if self.embeddings is not None and self.labels is not None:
            return self.embeddings, self.labels

        print("Computing embeddings...")

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=512,
            shuffle=False,
            num_workers=16,
            pin_memory=self.device == "cuda",
        )

        self.model.eval()
        self.model.to(self.device)

        embeddings = []
        labels = []
        current_samples = 0

        with torch.no_grad():
            for images, targets in dataloader:
                if self.samples is not None and current_samples >= self.samples:
                    break

                images = images.to(self.device)
                features = self.model(images)
                current_samples += len(images)

                embeddings.append(features.cpu())
                labels.extend(targets.tolist())

        self.embeddings = torch.cat(embeddings)
        self.labels = Tensor(labels)

        if self.samples is not None:
            self.embeddings = self.embeddings[: self.samples]
            self.labels = self.labels[: self.samples]

        return self.embeddings, self.labels

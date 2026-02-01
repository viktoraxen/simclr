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

        labels_np = labels.numpy()
        cmap = plt.cm.tab10
        unique_labels = sorted(set(labels_np.astype(int)))

        scatters = []
        for label in unique_labels:
            mask = labels_np == label
            sc = ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                projected[mask, 2],
                c=[cmap(label / 10)],
                alpha=0.6,
                s=10,
                label=CIFAR10_CLASSES[label],
            )
            scatters.append(sc)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
        ax.set_title("SimCLR Embeddings (3D t-SNE)")

        legend = ax.legend(loc="best", fontsize=12, markerscale=2)
        for legend_handle in legend.legend_handles:
            legend_handle.set_picker(True)
            legend_handle.set_pickradius(10)
        for text in legend.get_texts():
            text.set_picker(True)

        selected_class = [None]

        def on_pick(event):
            if event.artist in legend.legend_handles:
                idx = legend.legend_handles.index(event.artist)
            elif event.artist in legend.get_texts():
                idx = list(legend.get_texts()).index(event.artist)
            else:
                return

            if selected_class[0] == idx:
                for sc in scatters:
                    sc.set_alpha(0.6)

                selected_class[0] = None
            else:
                for i, sc in enumerate(scatters):
                    sc.set_alpha(0.6 if i == idx else 0.05)

                selected_class[0] = idx

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

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

from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    correct: int = 0
    total: int = 0


@dataclass
class ClassificationReport:
    overall: ClassificationMetrics
    by_class: dict[int, ClassificationMetrics]

    def print_table(self):
        console = Console()

        table = Table(title="Classification Report", title_style="normal")
        table.add_column("Class", justify="right", no_wrap=True)

        table.add_column("Accuracy", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("F1 Score", justify="right")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")

        for class_label, metrics in self.by_class.items():
            table.add_row(
                str(class_label),
                f"{metrics.accuracy * 100:.2f}%",
                f"{metrics.recall * 100:.2f}%",
                f"{metrics.precision * 100:.2f}%",
                f"{metrics.f1_score * 100:.2f}%",
                str(metrics.tp),
                str(metrics.total),
            )

        table.add_row(
            "Overall",
            f"{self.overall.accuracy * 100:.2f}%",
            f"{self.overall.recall * 100:.2f}%",
            f"{self.overall.precision * 100:.2f}%",
            f"{self.overall.f1_score * 100:.2f}%",
            str(self.overall.tp),
            str(self.overall.total),
            style="bold",
        )

        console.print(table)


class ClassificationTester(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    dataset: Dataset
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def metrics(self) -> ClassificationReport:
        dataloader = DataLoader(
            self.dataset,
            batch_size=len(self.dataset),
        )

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            inputs, labels = next(iter(dataloader))
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)

        correct_total = (preds == labels).sum().item()
        sample_total = labels.size(0)

        overall = ClassificationMetrics(
            accuracy=correct_total / sample_total,
            total=sample_total,
            correct=correct_total,
            tp=correct_total,
            fn=sample_total - correct_total,
        )

        by_class = {}

        for label_val in sorted(labels.unique()):
            val = label_val.item()
            m = ClassificationMetrics()

            actual = labels == val
            predicted = preds == val

            m.tp = (actual & predicted).sum().item()
            m.fp = (~actual & predicted).sum().item()
            m.fn = (actual & ~predicted).sum().item()
            m.tn = (~actual & ~predicted).sum().item()
            m.total = actual.sum().item()

            m.recall = m.tp / m.total if m.total > 0 else 0.0
            m.precision = m.tp / (m.tp + m.fp) if (m.tp + m.fp) > 0 else 0.0
            m.f1_score = (
                (2 * m.precision * m.recall / (m.precision + m.recall))
                if (m.precision + m.recall) > 0
                else 0.0
            )

            m.accuracy = m.recall

            by_class[val] = m

        return ClassificationReport(overall=overall, by_class=by_class)

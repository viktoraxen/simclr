from typing import Optional

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
)
from rich.table import Column
from rich.text import Text


class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def __init__(
        self,
        style: str = "progress.elapsed",
        compact: bool = False,
        table_column: Optional[Column] = None,
    ):
        super().__init__(table_column=table_column)
        self.compact = compact
        self.style = style

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed

        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")

        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style=self.style)


class EpochBar:
    """Context manager for epoch progress bar."""

    def __init__(self, epoch: int = 0, batches: int = 0):
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="grey50", finished_style="grey50"),
            TextColumn("[white]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(compact=False, style="grey50"),
            TextColumn("[grey50]Avg. loss: [white]{task.fields[current_average]:.4f}"),
            TextColumn("[grey50]Val. loss: [white]{task.fields[validation_loss]:.4f}"),
        ]
        self.progress = Progress(*columns)
        self.epoch = epoch
        self.batches = batches
        self.task = None

    def __enter__(self):
        self.progress.__enter__()
        self.task = self.progress.add_task(
            f"{self.epoch:>3}",
            total=self.batches,
            current_average=0.0,
            validation_loss=0.0,
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.__exit__(exc_type, exc_value, traceback)

    def update(self, average_loss: float, validation_loss: float = 0.0):
        """Update the progress bar with the latest average and validation loss."""
        if self.task is None:
            return

        self.progress.update(
            self.task,
            advance=1,
            current_average=average_loss,
            validation_loss=validation_loss,
        )

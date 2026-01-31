import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

from progress import EpochBar


class SimCLRTrainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    head: nn.Module
    optimizer: optim.Optimizer
    training_dataset: Dataset
    validation_dataset: Dataset
    temperature: float = 0.07
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _count_params(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def _format_params(self, count: int) -> str:
        if count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"

        if count >= 1_000:
            return f"{count / 1_000:.1f}K"

        return str(count)

    def _print_training_info(self, epochs: int, batch_size: int) -> None:
        model_params = self._count_params(self.model)
        head_params = self._count_params(self.head)
        total_params = model_params + head_params
        lr = self.optimizer.param_groups[0]["lr"]

        print(f"{'Model':<20} {type(self.model).__name__}")
        print(f"{'Model params':<20} {self._format_params(model_params)}")
        print(f"{'Head params':<20} {self._format_params(head_params)}")
        print(f"{'Total params':<20} {self._format_params(total_params)}")
        print(f"{'Training samples':<20} {len(self.training_dataset)}")
        print(f"{'Validation samples':<20} {len(self.validation_dataset)}")
        print(f"{'Batch size':<20} {batch_size}")
        print(f"{'Epochs':<20} {epochs}")
        print(f"{'Learning rate':<20} {lr}")
        print(f"{'Temperature':<20} {self.temperature}")
        print(f"{'Device':<20} {self.device}")
        print()

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 1,
    ):
        self._print_training_info(epochs, batch_size)

        training_loader = DataLoader(
            self.training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        average_validation_loss = 0

        self.model.to(self.device)
        self.head.to(self.device)

        for epoch in range(1, epochs + 1):
            total_loss = 0
            average_loss = 0.0

            self.model.train()
            self.head.train()

            with EpochBar(epoch, len(training_loader)) as epoch_bar:
                for i, inputs in enumerate(training_loader):
                    inputs = inputs.to(self.device, non_blocking=True).flatten(0, 1)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    outputs = self.head(outputs)
                    outputs = nn.functional.normalize(outputs, dim=1)

                    loss = self._loss_fn(outputs)
                    loss.backward()

                    self.optimizer.step()

                    total_loss += loss.item()
                    average_loss = total_loss / (i + 1)

                    epoch_bar.update(average_loss)

                self.model.eval()
                self.head.eval()

                total_validation_loss = 0
                average_validation_loss = 0.0

                with torch.no_grad():
                    for i, inputs in enumerate(validation_loader):
                        inputs = inputs.to(self.device, non_blocking=True).flatten(0, 1)

                        outputs = self.model(inputs)
                        outputs = self.head(outputs)
                        outputs = nn.functional.normalize(outputs, dim=1)
                        loss = self._loss_fn(outputs)

                        total_validation_loss += loss.item()
                        average_validation_loss = total_validation_loss / (i + 1)

                epoch_bar.update(average_loss, average_validation_loss)

        print(f"Training complete after {epochs} epochs.")

        return average_validation_loss

    def _loss_fn(self, outputs: Tensor) -> Tensor:
        batch_size = outputs.shape[0]

        logits = (outputs @ outputs.T) / self.temperature
        self_mask = torch.eye(batch_size, device=logits.device).bool()

        logits = logits.masked_fill(self_mask, float("-inf"))

        # Produces [1, 0, 3, 2, 5, 4, ...]
        labels = torch.arange(batch_size, device=logits.device)
        labels = torch.where(labels % 2 == 0, labels + 1, labels - 1)

        return nn.functional.cross_entropy(logits, labels)

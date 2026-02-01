import copy
from typing import Any, Callable

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
    scheduler: Any
    training_dataset: Dataset
    validation_dataset: Dataset
    patience: int | None = None
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
        weight_decay = self.optimizer.param_groups[0].get("weight_decay", 0)

        print(f"{'Model':<20} {type(self.model).__name__}")
        print(f"{'Model params':<20} {self._format_params(model_params)}")
        print(f"{'Head params':<20} {self._format_params(head_params)}")
        print(f"{'Total params':<20} {self._format_params(total_params)}")
        print()
        print(f"{'Training samples':<20} {len(self.training_dataset)}")
        print(f"{'Validation samples':<20} {len(self.validation_dataset)}")
        print(f"{'Batch size':<20} {batch_size}")
        print(f"{'Max epochs':<20} {epochs}")
        print()
        print(f"{'Optimizer':<20} {type(self.optimizer).__name__}")
        print(f"{'Learning rate':<20} {lr}")
        print(f"{'Weight decay':<20} {weight_decay}")
        print(f"{'Scheduler':<20} {type(self.scheduler).__name__}")
        print(f"{'Early stop patience':<20} {self.patience}")
        print()
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
            pin_memory=self.device == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )

        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=self.device == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )

        average_validation_loss = 0.0
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        final_epoch = epochs

        self.model.to(self.device)
        self.head.to(self.device)

        best_model_state = copy.deepcopy(self.model.state_dict())
        best_head_state = copy.deepcopy(self.head.state_dict())

        for epoch in range(1, epochs + 1):
            total_loss = 0
            average_loss = 0.0

            self.model.train()
            self.head.train()

            with EpochBar(epoch, len(training_loader)) as epoch_bar:
                for i, inputs in enumerate(training_loader):
                    inputs = torch.stack(inputs, dim=1).flatten(0, 1)
                    inputs = inputs.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    outputs = self.head(outputs)

                    # Normalize and compute loss in float32 to avoid overflow
                    outputs = nn.functional.normalize(outputs.float(), dim=1)
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
                        inputs = torch.stack(inputs, dim=1).flatten(0, 1)
                        inputs = inputs.to(self.device, non_blocking=True)

                        outputs = self.model(inputs)
                        outputs = self.head(outputs)

                        outputs = nn.functional.normalize(outputs.float(), dim=1)
                        loss = self._loss_fn(outputs)

                        total_validation_loss += loss.item()
                        average_validation_loss = total_validation_loss / (i + 1)

                epoch_bar.update(average_loss, average_validation_loss)

            if average_validation_loss < best_validation_loss:
                epochs_without_improvement = 0
                best_validation_loss = average_validation_loss

                best_model_state = copy.deepcopy(self.model.state_dict())
                best_head_state = copy.deepcopy(self.head.state_dict())
            else:
                epochs_without_improvement += 1

            self.scheduler.step()

            if self.patience is not None and epochs_without_improvement > self.patience:
                final_epoch = epoch
                break

        self.model.load_state_dict(best_model_state)
        self.head.load_state_dict(best_head_state)

        print(f"Training complete after {final_epoch} epochs.")

        return best_validation_loss

    def _loss_fn(self, outputs: Tensor) -> Tensor:
        batch_size = outputs.shape[0]

        logits = (outputs @ outputs.T) / self.temperature
        self_mask = torch.eye(batch_size, device=logits.device).bool()

        logits = logits.masked_fill(self_mask, float("-inf"))

        # Produces [1, 0, 3, 2, 5, 4, ...]
        labels = torch.arange(batch_size, device=logits.device)
        labels = torch.where(labels % 2 == 0, labels + 1, labels - 1)

        return nn.functional.cross_entropy(logits, labels)


class ClassifierTrainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Any
    training_dataset: Dataset
    validation_dataset: Dataset
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def train(
        self,
        batch_size: int = 1,
        epochs: int = 1,
    ):
        training_loader = DataLoader(
            self.training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=self.device == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )

        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=self.device == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )

        average_validation_loss = 0.0
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        final_epoch = epochs

        self.model.to(self.device)

        best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, epochs + 1):
            total_loss = 0
            average_loss = 0.0

            self.model.train()

            with EpochBar(epoch, len(training_loader)) as epoch_bar:
                for i, (inputs, labels) in enumerate(training_loader):
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)

                    loss = self.loss_fn(outputs, labels)
                    loss.backward()

                    self.optimizer.step()

                    total_loss += loss.item()
                    average_loss = total_loss / (i + 1)

                    epoch_bar.update(average_loss)

                self.model.eval()

                total_validation_loss = 0
                average_validation_loss = 0.0

                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(validation_loader):
                        inputs = inputs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, labels)

                        total_validation_loss += loss.item()
                        average_validation_loss = total_validation_loss / (i + 1)

                epoch_bar.update(average_loss, average_validation_loss)

            if average_validation_loss < best_validation_loss:
                epochs_without_improvement = 0
                best_validation_loss = average_validation_loss

                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1

            self.scheduler.step(average_validation_loss)

            if epochs_without_improvement > self.patience:
                final_epoch = epoch
                break

        self.model.load_state_dict(best_model_state)

        print(f"Training complete after {final_epoch} epochs.")

        return best_validation_loss

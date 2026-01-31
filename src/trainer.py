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

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 1,
    ):
        print(f"Training using device '{self.device}'.")

        training_loader = DataLoader(
            self.training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
        )

        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            num_workers=16,
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
                    inputs = inputs.to(self.device).flatten(0, 1)

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
                        inputs = inputs.to(self.device).flatten(0, 1)

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

    def _unpack_data(self, data):
        inputs, labels = data

        return inputs.to(self.device), labels.to(self.device)

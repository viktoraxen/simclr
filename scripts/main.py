from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import SimCLRDataset
from network import ResNet6
from trainer import SimCLRTrainer


def main():
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-6

    dataset_train = SimCLRDataset(train=True)
    dataset_test = SimCLRDataset(train=False)

    model = ResNet6()
    head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )

    optimizer = AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    trainer = SimCLRTrainer(
        model=model,
        head=head,
        training_dataset=dataset_train,
        validation_dataset=dataset_test,
        optimizer=optimizer,
    )

    loss = trainer.train(
        epochs=50,
        batch_size=batch_size,
    )

    model_name = type(model).__name__
    result_name = f"{model_name}_{loss:.4f}"
    result_path = f"models/{result_name}.pth"

    q = input(f"Save to '{result_path}'? Y/n:")

    if q == "y" or q == "":
        Path("models").mkdir(exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "head": head.state_dict()},
            result_path,
        )
        print(f"Model saved to '{result_path}'.")


if __name__ == "__main__":
    main()

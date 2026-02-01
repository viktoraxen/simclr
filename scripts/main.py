from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)

from dataset import SimCLRDataset
from network import ResNet9
from trainer import SimCLRTrainer


def main():
    epochs = 1000
    batch_size = 512
    learning_rate = 0.3 * batch_size / 256
    weight_decay = 1e-6
    scheduler_patience = 10
    trainer_patience = 30

    dataset_train = SimCLRDataset(train=True)
    dataset_test = SimCLRDataset(train=False)

    model = ResNet9()
    head = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )

    optimizer = AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.3,
        patience=scheduler_patience,
        min_lr=0,
    )

    trainer = SimCLRTrainer(
        model=model,
        head=head,
        training_dataset=dataset_train,
        validation_dataset=dataset_test,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=trainer_patience,
    )

    loss = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
    )

    model_name = type(model).__name__
    result_name = f"{model_name}_{loss:.4f}"
    result_path = f"models/{result_name}.pth"

    # q = input(f"Save to '{result_path}'? Y/n:")
    #
    # if q == "y" or q == "":

    Path("models").mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "head": head.state_dict()},
        result_path,
    )
    print(f"Model saved to '{result_path}'.")


if __name__ == "__main__":
    main()

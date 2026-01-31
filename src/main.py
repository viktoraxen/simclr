import torch.nn as nn
from torch.optim import Adam

from dataset import SimCLRDataset
from network import ResNet6
from trainer import SimCLRTrainer


def main():
    dataset_train = SimCLRDataset(train=True)
    dataset_test = SimCLRDataset(train=False)

    resnet = ResNet6()
    head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )

    optimizer = Adam(
        list(resnet.parameters()) + list(head.parameters()),
        lr=1e-3,
    )

    trainer = SimCLRTrainer(
        model=resnet,
        head=head,
        training_dataset=dataset_train,
        validation_dataset=dataset_test,
        optimizer=optimizer,
    )

    trainer.train(
        epochs=15,
        batch_size=128,
    )


if __name__ == "__main__":
    main()

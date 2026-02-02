from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import CIFAR10Extracted
from network import init_model
from tester import ClassificationTester
from trainer import ClassifierTrainer


def main():
    epochs = 100
    batch_size = 1024
    learning_rate = 1e-3
    weight_decay = 1e-6
    scheduler_patience = 10
    scheduler_reduce_factor = 0.1
    trainer_patience = 30

    extrator = init_model("models")
    training_dataset = CIFAR10Extracted(extrator, train=True)
    validation_dataset = CIFAR10Extracted(extrator, train=False)

    model = nn.Sequential(
        nn.Linear(192, 192),
        nn.ReLU(),
        nn.Linear(192, 10),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=scheduler_reduce_factor,
        patience=scheduler_patience,
    )

    trainer = ClassifierTrainer(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.CrossEntropyLoss(),
        patience=trainer_patience,
    )

    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
    )

    tester = ClassificationTester(
        model=model,
        dataset=validation_dataset,
    )

    metrics = tester.metrics()
    metrics.print_table()


if __name__ == "__main__":
    main()

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import SimCLRDataset
from network import ResNet10Wide, save_model
from trainer import SimCLRTrainer


def main():
    epochs = 600
    batch_size = 512 + 128
    learning_rate = 0.3 * batch_size / 256
    weight_decay = 1e-6
    warmup_epochs = 30
    temperature = 0.07

    dataset_train = SimCLRDataset(train=True)
    dataset_test = SimCLRDataset(train=False)

    model = ResNet10Wide()
    head = nn.Sequential(
        nn.Linear(192, 384),
        nn.ReLU(),
        nn.Linear(384, 192),
    )

    optimizer = AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    trainer = SimCLRTrainer(
        model=model,
        head=head,
        training_dataset=dataset_train,
        validation_dataset=dataset_test,
        optimizer=optimizer,
        scheduler=scheduler,
        temperature=temperature,
    )

    loss = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
    )

    save_model(model, "models", f"_{loss:.4f}")


if __name__ == "__main__":
    main()

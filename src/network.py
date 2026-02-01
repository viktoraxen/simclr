from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        activation: bool = True,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU() if activation else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()

        if in_channels == out_channels:
            stride = 1

            self.residual = nn.Identity()
        elif (in_channels * 2) == out_channels:
            stride = 2

            self.residual = ConvLayer(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=1,
                activation=False,
            )
        else:
            raise ValueError("Out channels must be equal to or double that of in channels!")

        layers = [ConvLayer(in_channels, out_channels, stride=stride)]

        for i in range(num_layers - 1):
            layers.append(
                ConvLayer(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    activation=(i != num_layers - 2),
                ),
            )

        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.body(x)
        x_ += self.residual(x)

        return F.relu(x_)


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        # In 3, 32, 32
        # Out 32, 32, 32
        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
        )

        # In 32, 32, 32
        # Out 32, 32, 32
        self.conv32 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )

        # In 32, 32, 32
        # Out 64, 16, 16
        self.conv64 = nn.Sequential(
            BasicBlock(32, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

        # In 64, 16, 16
        # Out 128, 8, 8
        self.conv128 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        # In 32, 8, 8
        # Out 128
        self.body = nn.Sequential(
            self.conv32,
            self.conv64,
            self.conv128,
            nn.AvgPool2d(8),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.tail(x)
        x = self.body(x)

        return x

    @classmethod
    def load(cls, path: str | Path, device: str) -> "ResNet9":
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        model = cls()
        model.load_state_dict(checkpoint)

        return model


def save_model(model: nn.Module, path: str | Path, suffix: str = ""):
    model_name = type(model).__name__
    model_path = Path(path) / f"{model_name}{suffix}.pth"
    model_path.parent.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        model_path,
    )

    print(f"Model saved to '{model_path}'.")


def init_model(models_dir: str | Path | None = None) -> nn.Module:
    if models_dir is None:
        return ResNet9()

    models_dir = Path(models_dir) if isinstance(models_dir, str) else models_dir
    models = sorted(models_dir.glob("*.pth")) if models_dir.exists() else []

    if not models:
        return ResNet9()

    print("Available models:")

    print("  0: Untrained")

    for i, model_path in enumerate(models):
        print(f"  {i + 1}: {model_path.name}")

    choice = int(input("\nSelect model: "))

    while not (0 <= choice <= len(models)):
        choice = int(input("\nInvalid selection, try again: "))

    if choice == 0:
        return ResNet9()

    return ResNet9.load(
        models[choice - 1],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

## SimCLR

This repository contains experiments training a custom ResNet-based architecture using the SimCLR framework to create meaningful image embeddings.

### Running the project

Install dependencies using `uv`:

```bash
# If using GPU
uv sync

# If using CPU
uv sync --extra cpu
```

Train the backbone (trained model is saved to `models/ResNet9_<best_validation_loss>.pth`):

```bash
uv run scripts/main.py
```

Train and evaluate the classification head:

```bash
uv run scripts/classifier.py
```

To see a t-SNE projection of the backbones embeddings:

```bash
uv run scripts/visualize.py
```

## Experiments

Description of performed experiments and results.

### Dataset

Training was done using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The images are converted to tensors with shape `[3, 32, 32]`.

### Network

The ResNet9 network used is defined in `src/network.py`.

It starts with an upscaling to 32 feature dimensions using a 5x5 convolutional kernel, followed by three sets of three basic convolution blocks with residual connections. The last set outputs a feature map of shape `[128, 8, 8]`, which is converted to a `128`-dimensional feature vector through average pooling.

### Evaluation

Evaluation was done by training a small classification head with one 128 neuron wide hidden layer, with a ReLU activation, trained using the cross-entropy loss.

### Results

Best achieved accuracy (so far) by the classifier head is **76.81 %**

The used backbone ResNet had a contrastive loss of **1.1987**

#### Visualizations

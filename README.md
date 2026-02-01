## SimCLR Experiments

This repository contains experiments training a custom ResNet-based architecture using the SimCLR framework to create meaningful image embeddings.

### Dataset

Training was done using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The images are converted to tensors with shape `[3, 32, 32]`.

### Network

The ResNet9 network used is defined in `src/network.py`.

The network starts with an upscaling to 32 feature dimensions using a 5x5 convolutional kernel, followed by three sets of three basic convolution blocks with residual connections. The last set outputs a feature map of shape `[128, 8, 8]`, which is converted to a `128`-dimensional feature vector through average pooling.

### Evaluation

The trained network is evaluated by training a small classification head using the cross-entropy loss. The classification head is defined in pytorch:

```python
model = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

### Results

The currently best achieved accuracy by the classifier head is **76.81 %**

The used ResNet had a contrastive loss of **1.1987**

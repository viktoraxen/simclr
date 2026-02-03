Model                ResNet10
Model params         1.38M
Head params          33.0K
Total params         1.42M

Training samples     50000
Validation samples   10000
Batch size           512
Max epochs           1000

Optimizer            AdamW
Learning rate        0.6
Weight decay         1e-06
Scheduler            ReduceLROnPlateau
Early stop patience  30

Temperature          0.07
Device               cuda

Validation loss: 1.1987
Accuracy: 76.81 %

Model                ResNet10
Model params         1.38M
Head params          65.9K
Total params         1.45M

Training samples     50000
Validation samples   10000
Batch size           1024
Max epochs           400

Optimizer            AdamW
Learning rate        0.012
Weight decay         1e-06
Scheduler            SequentialLR
Early stop patience  None

Temperature          0.07
Device               cuda

Validation loss: 1.3703
Accuracy: 80.72 %

Model                ResNet10Wide
Model params         3.11M
Head params          148.0K
Total params         3.26M

Training samples     50000
Validation samples   10000
Batch size           640
Max epochs           600

Optimizer            AdamW
Learning rate        0.0075
Weight decay         1e-06
Scheduler            SequentialLR
Early stop patience  None

Temperature          0.07
Device               cuda

Validation loss: 0.9199
Accuracy: 85.16 %

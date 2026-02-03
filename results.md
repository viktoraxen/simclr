## Model

| Parameter    | Run 1    | Run 2    | Run 3        |
|--------------|----------|----------|--------------|
| Model        | ResNet10 | ResNet10 | ResNet10Wide |
| Model params | 1.38M    | 1.38M    | 3.11M        |
| Head params  | 33.0K    | 65.9K    | 148.0K       |
| Total params | 1.42M    | 1.45M    | 3.26M        |

## Training

| Parameter           | Run 1              | Run 2        | Run 3        |
|---------------------|--------------------|--------------|--------------|
| Training samples    | 50000              | 50000        | 50000        |
| Validation samples  | 10000              | 10000        | 10000        |
| Batch size          | 512                | 1024         | 640          |
| Max epochs          | 1000               | 400          | 600          |
| Optimizer           | AdamW              | AdamW        | AdamW        |
| Learning rate       | 0.6                | 0.012        | 0.0075       |
| Weight decay        | 1e-06              | 1e-06        | 1e-06        |
| Scheduler           | ReduceLROnPlateau  | SequentialLR | SequentialLR |
| Early stop patience | 30                 | None         | None         |
| Temperature         | 0.07               | 0.07         | 0.07         |
| Device              | cuda               | cuda         | cuda         |

## Results

| Metric          | Run 1  | Run 2  | Run 3  |
|-----------------|--------|--------|--------|
| Validation loss | 1.1987 | 1.3703 | 0.9199 |
| Accuracy        | 76.81% | 80.72% | 85.16% |

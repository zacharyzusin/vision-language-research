# Evaluation Metrics Module

This module provides comprehensive evaluation metrics that can be used across all model types.

## Metrics Computed

1. **Top-1 Accuracy (overall)**: Standard accuracy across all samples
2. **Balanced Accuracy (per-class)**: Average accuracy per class (handles class imbalance)
3. **Average Margin**: Difference between top and runner-up prediction confidence
4. **Many-shot Accuracy**: Classes with >100 training examples
5. **Medium-shot Accuracy**: Classes with 20-100 training examples
6. **Few-shot Accuracy**: Classes with <20 training examples

## Usage

### Basic Usage

```python
from src.metrics.evaluation_metrics import (
    compute_class_shots,
    compute_metrics,
    print_metrics
)

# 1. Compute class shots from training set
train_labels = ...  # Tensor of training labels
num_classes = ...   # Number of classes
class_shots = compute_class_shots(train_labels, num_classes)

# 2. Get model logits and labels from validation set
val_logits = ...    # Tensor of shape (N, num_classes)
val_labels = ...    # Tensor of shape (N,)

# 3. Compute metrics
metrics = compute_metrics(
    logits=val_logits,
    labels=val_labels,
    class_shots=class_shots,  # Optional, but needed for shot-based metrics
    device=device
)

# 4. Print formatted results
print_metrics(metrics, title="My Model Evaluation")
```

### Example: Adding Metrics to a New Model

```python
import torch
from src.metrics import compute_class_shots, compute_metrics, print_metrics

# Load your model and datasets
model = ...
train_ds = ...
val_ds = ...

# Compute class shots
train_loader = DataLoader(train_ds, ...)
train_labels = []
for _, labels in train_loader:
    train_labels.append(labels)
train_labels = torch.cat(train_labels, dim=0)
class_shots = compute_class_shots(train_labels, train_ds.num_classes)

# Evaluate model
val_loader = DataLoader(val_ds, ...)
all_logits = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        logits = model(images)  # Your model's forward pass
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

logits = torch.cat(all_logits, dim=0)
labels = torch.cat(all_labels, dim=0)

# Compute and print metrics
metrics = compute_metrics(logits, labels, class_shots=class_shots, device=device)
print_metrics(metrics, title="My New Model Evaluation")
```

## Functions

### `compute_class_shots(train_labels, num_classes)`
Computes the number of training examples per class.

**Args:**
- `train_labels`: Tensor of training labels
- `num_classes`: Number of classes

**Returns:**
- Dictionary mapping `class_id -> number of examples`

### `compute_metrics(logits, labels, class_shots=None, device=None)`
Computes all evaluation metrics.

**Args:**
- `logits`: Model logits tensor of shape (N, num_classes)
- `labels`: Ground truth labels tensor of shape (N,)
- `class_shots`: Optional dictionary mapping class_id -> number of training examples
- `device`: Optional device (defaults to logits.device)

**Returns:**
- Dictionary of metric names -> values

### `print_metrics(metrics, title="Evaluation Metrics")`
Prints metrics in a formatted way.

**Args:**
- `metrics`: Dictionary from `compute_metrics()`
- `title`: Optional title for the metrics section

## Integration

This module is already integrated into:
- `clip_zeroshot_baseline.py` - Zero-shot CLIP evaluation
- `clip_linear_probe.py` - Linear probe evaluation
- `eval.py` - Fine-tuned model evaluation

To add to a new model, simply import and use the functions as shown above.


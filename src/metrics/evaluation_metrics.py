"""
Shared evaluation metrics for all models.

This module provides comprehensive evaluation metrics that can be used
across different model types (zero-shot CLIP, linear probe, fine-tuned models, etc.).
"""

import torch
import torch.nn.functional as F
from collections import Counter
from typing import Dict, Optional, Union
from torch.utils.data import Dataset
from tqdm import tqdm


def get_labels_from_dataset(dataset: Dataset) -> torch.Tensor:
    """
    Efficiently extract all labels from a dataset without loading images.
    
    This function tries to access labels directly from dataset internals,
    falling back to iterating through the dataset if needed.
    
    Args:
        dataset: PyTorch Dataset instance
    
    Returns:
        Tensor of labels
    """
    # Try to access labels directly from dataset internals (fastest)
    if hasattr(dataset, 'samples'):
        # Most datasets store (path, label) or (idx, label) tuples
        samples = dataset.samples
        if len(samples) > 0 and isinstance(samples[0], (tuple, list)):
            # Extract labels (second element of each tuple)
            labels = [sample[1] for sample in samples]
            return torch.tensor(labels, dtype=torch.long)
    
    # For SubsetDataset, try to access base dataset
    if hasattr(dataset, 'base_dataset'):
        return get_labels_from_dataset(dataset.base_dataset)
    
    # Fallback: iterate through dataset (slower but works for all datasets)
    # Use a small batch to avoid loading all images
    labels = []
    for i in range(len(dataset)):
        try:
            _, label = dataset[i]
            labels.append(int(label))
        except Exception:
            # Skip problematic samples
            continue
    
    return torch.tensor(labels, dtype=torch.long)


def compute_class_shots(
    train_labels: Union[torch.Tensor, Dataset], 
    num_classes: Optional[int] = None
) -> Dict[int, int]:
    """
    Compute number of training examples per class.
    
    This function can accept either:
    - A tensor of labels (fast)
    - A Dataset instance (will extract labels efficiently)
    
    Args:
        train_labels: Training labels tensor OR Dataset instance
        num_classes: Number of classes (required if train_labels is Dataset)
    
    Returns:
        Dictionary mapping class_id -> number of examples
    """
    # If dataset is provided, extract labels efficiently
    if isinstance(train_labels, Dataset):
        if num_classes is None:
            if hasattr(train_labels, 'num_classes'):
                num_classes = train_labels.num_classes
            else:
                raise ValueError("num_classes must be provided when train_labels is a Dataset")
        train_labels = get_labels_from_dataset(train_labels)
    
    # Convert to list for counting
    if isinstance(train_labels, torch.Tensor):
        labels_list = train_labels.cpu().tolist()
    else:
        labels_list = train_labels
    
    class_counts = Counter(labels_list)
    
    # Determine num_classes if not provided
    if num_classes is None:
        num_classes = max(labels_list) + 1 if labels_list else 0
    
    # Ensure all classes are represented (even if 0)
    class_shots = {i: class_counts.get(i, 0) for i in range(num_classes)}
    return class_shots


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_shots: Optional[Dict[int, int]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 10000,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        logits: Model logits (N, num_classes)
        labels: Ground truth labels (N,)
        class_shots: Optional dictionary mapping class_id -> number of training examples
        device: Optional device (if None, uses CPU to avoid OOM)
        batch_size: Batch size for processing large tensors (default: 10000)
    
    Returns:
        Dictionary of metric names -> values
    """
    # Use CPU by default to avoid GPU OOM for large validation sets
    # Only use GPU if explicitly requested and logits are already on GPU
    if device is None:
        # If logits are on CPU, keep on CPU. If on GPU, use GPU but warn about potential OOM
        if logits.device.type == "cuda":
            device = logits.device
        else:
            device = torch.device("cpu")
    
    # Move to device in batches to avoid OOM
    num_samples = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Initialize accumulators
    total_correct = 0
    per_class_correct = torch.zeros(num_classes, device=device)
    per_class_total = torch.zeros(num_classes, device=device)
    total_margin = 0.0
    
    # Process in batches to avoid OOM (with progress bar for large datasets)
    num_batches = (num_samples + batch_size - 1) // batch_size
    use_progress = num_samples > 10000  # Show progress for large datasets
    
    iterator = range(0, num_samples, batch_size)
    if use_progress:
        iterator = tqdm(iterator, desc="Computing metrics", total=num_batches, unit="batch")
    
    for i in iterator:
        end_idx = min(i + batch_size, num_samples)
        batch_logits = logits[i:end_idx].to(device)
        batch_labels = labels[i:end_idx].to(device)
        
        # Get predictions
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_preds = torch.argmax(batch_logits, dim=1)
        
        # 1. Top-1 accuracy (accumulate correct predictions)
        total_correct += (batch_preds == batch_labels).sum().item()
        
        # 2. Per-class accuracy (accumulate) - optimized for speed
        # Only process classes that actually appear in this batch
        unique_labels = torch.unique(batch_labels)
        for c in unique_labels:
            c = c.item()
            mask = batch_labels == c
            per_class_correct[c] += (batch_preds[mask] == batch_labels[mask]).sum()
            per_class_total[c] += mask.sum()
        
        # 3. Margin (accumulate)
        sorted_probs, _ = torch.sort(batch_probs, dim=1, descending=True)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]  # top - runner-up
        total_margin += margins.sum().item()
    
    # 1. Top-1 accuracy (overall)
    top1_acc = total_correct / num_samples
    
    # 2. Balanced accuracy (average per-class accuracy)
    valid_classes = per_class_total > 0
    if valid_classes.sum() > 0:
        per_class_acc = per_class_correct[valid_classes] / per_class_total[valid_classes]
        balanced_acc = per_class_acc.mean().item()
    else:
        balanced_acc = 0.0
    
    # 3. Average margin
    avg_margin = total_margin / num_samples
    
    # 4-6. Accuracy by shot category (if class_shots provided)
    many_shot_acc = 0.0
    medium_shot_acc = 0.0
    few_shot_acc = 0.0
    many_shot_count = 0
    medium_shot_count = 0
    few_shot_count = 0
    num_many_shot_classes = 0
    num_medium_shot_classes = 0
    num_few_shot_classes = 0
    
    # 4-6. Accuracy by shot category (if class_shots provided)
    # Process shot-based metrics in batches as well
    if class_shots is not None:
        # Categorize classes by number of training examples
        many_shot_classes = [c for c, count in class_shots.items() if count > 100]
        medium_shot_classes = [c for c, count in class_shots.items() if 20 <= count <= 100]
        few_shot_classes = [c for c, count in class_shots.items() if count < 20]
        
        num_many_shot_classes = len(many_shot_classes)
        num_medium_shot_classes = len(medium_shot_classes)
        num_few_shot_classes = len(few_shot_classes)
        
        # Initialize shot category accumulators
        many_shot_correct = 0
        many_shot_total = 0
        medium_shot_correct = 0
        medium_shot_total = 0
        few_shot_correct = 0
        few_shot_total = 0
        
        # Convert to tensors for efficient masking
        many_shot_tensor = torch.tensor(many_shot_classes, device=device) if many_shot_classes else None
        medium_shot_tensor = torch.tensor(medium_shot_classes, device=device) if medium_shot_classes else None
        few_shot_tensor = torch.tensor(few_shot_classes, device=device) if few_shot_classes else None
        
        # Process in batches (reuse same batches, no need for separate progress bar)
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_logits = logits[i:end_idx].to(device)
            batch_labels = labels[i:end_idx].to(device)
            batch_preds = torch.argmax(batch_logits, dim=1)
            
            def compute_shot_acc_batch(class_tensor):
                if class_tensor is None or len(class_tensor) == 0:
                    return 0, 0
                class_mask = torch.isin(batch_labels, class_tensor)
                if class_mask.sum() == 0:
                    return 0, 0
                correct = (batch_preds[class_mask] == batch_labels[class_mask]).sum().item()
                total = class_mask.sum().item()
                return correct, total
            
            m_correct, m_total = compute_shot_acc_batch(many_shot_tensor)
            many_shot_correct += m_correct
            many_shot_total += m_total
            
            med_correct, med_total = compute_shot_acc_batch(medium_shot_tensor)
            medium_shot_correct += med_correct
            medium_shot_total += med_total
            
            f_correct, f_total = compute_shot_acc_batch(few_shot_tensor)
            few_shot_correct += f_correct
            few_shot_total += f_total
        
        many_shot_acc = many_shot_correct / many_shot_total if many_shot_total > 0 else 0.0
        medium_shot_acc = medium_shot_correct / medium_shot_total if medium_shot_total > 0 else 0.0
        few_shot_acc = few_shot_correct / few_shot_total if few_shot_total > 0 else 0.0
        many_shot_count = many_shot_total
        medium_shot_count = medium_shot_total
        few_shot_count = few_shot_total
    
    return {
        "top1_acc": top1_acc,
        "balanced_acc": balanced_acc,
        "avg_margin": avg_margin,
        "many_shot_acc": many_shot_acc,
        "medium_shot_acc": medium_shot_acc,
        "few_shot_acc": few_shot_acc,
        "many_shot_count": many_shot_count,
        "medium_shot_count": medium_shot_count,
        "few_shot_count": few_shot_count,
        "num_many_shot_classes": num_many_shot_classes,
        "num_medium_shot_classes": num_medium_shot_classes,
        "num_few_shot_classes": num_few_shot_classes,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics()
        title: Title for the metrics section
    """
    print("\n" + "="*70)
    print(title + ":")
    print("="*70)
    print(f"Top-1 Accuracy (overall):        {metrics['top1_acc']*100:.2f}%")
    print(f"Balanced Accuracy (per-class):     {metrics['balanced_acc']*100:.2f}%")
    print(f"Average Margin:                   {metrics['avg_margin']:.4f}")
    print()
    
    if metrics['num_many_shot_classes'] > 0 or metrics['num_medium_shot_classes'] > 0 or metrics['num_few_shot_classes'] > 0:
        print("Accuracy by Shot Category:")
        if metrics['many_shot_count'] > 0:
            print(f"  Many-shot (>100 examples):     {metrics['many_shot_acc']*100:.2f}% "
                  f"({metrics['many_shot_count']} samples, {metrics['num_many_shot_classes']} classes)")
        if metrics['medium_shot_count'] > 0:
            print(f"  Medium-shot (20-100 examples): {metrics['medium_shot_acc']*100:.2f}% "
                  f"({metrics['medium_shot_count']} samples, {metrics['num_medium_shot_classes']} classes)")
        if metrics['few_shot_count'] > 0:
            print(f"  Few-shot (<20 examples):        {metrics['few_shot_acc']*100:.2f}% "
                  f"({metrics['few_shot_count']} samples, {metrics['num_few_shot_classes']} classes)")
    
    print("="*70)


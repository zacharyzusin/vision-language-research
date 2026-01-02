"""
Linear Probe on frozen CLIP features.

This script trains a linear classifier on top of frozen CLIP image features
to establish a baseline for comparison with fine-tuning approaches.

Supports:
- iNaturalist (2018/2021)
- Stanford Cars
- KikiBouba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import os
import argparse
from typing import Literal
import numpy as np

# Import dataset loaders
from src.datasets.inat_dataset import get_inat
from src.datasets.stanford_cars_dataset import get_stanford_cars
from src.datasets.kikibouba_dataset import get_kikibouba
from src.metrics.evaluation_metrics import compute_class_shots, compute_metrics, print_metrics


class LinearProbe(nn.Module):
    """Simple linear classifier on top of frozen features."""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features):
        return self.classifier(features)


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Extract CLIP image features for all samples in a dataloader.
    
    Args:
        model: CLIP model (frozen)
        dataloader: DataLoader with images
        device: Device to run on
    
    Returns:
        features: Tensor of shape (N, feature_dim)
        labels: Tensor of shape (N,)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract image features
        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        
        all_features.append(image_features.cpu())
        all_labels.append(labels.cpu())
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    verbose: bool = True,
):
    """
    Train a linear classifier on frozen features.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        num_classes: Number of classes
        device: Device to train on
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        weight_decay: Weight decay (L2 regularization)
    
    Returns:
        Tuple of (best_model, best_val_acc)
    """
    feature_dim = train_features.shape[1]
    
    # Create model
    model = LinearProbe(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Already in memory
        pin_memory=True,
    )
    
    best_val_acc = 0.0
    best_model_state = None
    
    if verbose:
        print(f"\nTraining linear probe for {epochs} epochs...")
        print(f"  Training samples: {len(train_features)}")
        print(f"  Validation samples: {len(val_features)}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Number of classes: {num_classes}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_features_gpu = val_features.to(device)
            val_labels_gpu = val_labels.to(device)
            val_logits = model(val_features_gpu)
            val_preds = torch.argmax(val_logits, dim=1)
            val_correct = (val_preds == val_labels_gpu).sum().item()
            val_total = val_labels_gpu.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={avg_loss:.4f}, "
                f"Train Acc={train_acc*100:.2f}%, "
                f"Val Acc={val_acc*100:.2f}%"
            )
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Linear Probe on frozen CLIP features"
    )
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--dataset", type=str, default="inat",
                       choices=["inat", "stanford_cars", "kikibouba"],
                       help="Dataset type")
    parser.add_argument("--version", type=str, default="2021",
                       help="Dataset version (for iNaturalist: 2018 or 2021)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name (e.g., ViT-B/16, ViT-B/32)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for feature extraction")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate for linear probe")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--probe_batch_size", type=int, default=256,
                       help="Batch size for training linear probe")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "inat":
        print(f"  Version: {args.version}")
    
    # Load CLIP model
    print(f"\nLoading CLIP model: {args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()
    # Freeze CLIP model
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        dummy_features = clip_model.encode_image(dummy_input)
        feature_dim = dummy_features.shape[1]
    print(f"CLIP feature dimension: {feature_dim}")
    
    # Load datasets
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "inat":
        train_ds = get_inat(args.data_root, "train", version=args.version)
        val_ds = get_inat(args.data_root, "val", version=args.version)
        num_classes = train_ds.num_classes
    elif args.dataset == "stanford_cars":
        train_ds = get_stanford_cars(args.data_root, "train")
        val_ds = get_stanford_cars(args.data_root, "val")
        num_classes = train_ds.num_classes
    elif args.dataset == "kikibouba":
        train_ds = get_kikibouba(args.data_root, "train")
        val_ds = get_kikibouba(args.data_root, "val")
        num_classes = train_ds.num_classes
    
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    
    # Create data loaders for feature extraction
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # Don't need to shuffle for feature extraction
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Extract features
    print("\nExtracting training features...")
    train_features, train_labels = extract_features(clip_model, train_loader, device)
    
    print("\nExtracting validation features...")
    val_features, val_labels = extract_features(clip_model, val_loader, device)
    
    # Compute class shots from training set (already have labels from feature extraction)
    print("\nComputing class shot statistics from training set...")
    class_shots = compute_class_shots(train_labels, num_classes)
    
    # Count classes by shot category
    many_shot = sum(1 for count in class_shots.values() if count > 100)
    medium_shot = sum(1 for count in class_shots.values() if 20 <= count <= 100)
    few_shot = sum(1 for count in class_shots.values() if count < 20)
    print(f"  Class distribution: {many_shot} many-shot (>100), {medium_shot} medium-shot (20-100), {few_shot} few-shot (<20)")
    
    # Train linear probe
    best_model, best_val_acc = train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_classes=num_classes,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.probe_batch_size,
        weight_decay=args.weight_decay,
    )
    
    # Evaluate best model to get logits for comprehensive metrics
    print("\nComputing comprehensive metrics with best model...")
    best_model.eval()
    with torch.no_grad():
        val_features_gpu = val_features.to(device)
        val_labels_gpu = val_labels.to(device)
        val_logits = best_model(val_features_gpu)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(
        val_logits.cpu(), 
        val_labels.cpu(), 
        class_shots=class_shots, 
        device=device
    )
    
    # Print results
    print_metrics(metrics, title="Linear Probe Evaluation")


if __name__ == "__main__":
    main()


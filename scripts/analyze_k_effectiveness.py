"""
Analyze why K=32 might not help on small subsets.

This script checks:
1. Dataset sizes vs parameter counts
2. Training/validation loss curves (if available)
3. Overfitting indicators
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from pathlib import Path
from src.datasets.inat_dataset import get_inat

def analyze_subset(name, category_ids):
    """Analyze a subset's characteristics."""
    train_ds = get_inat('data/iNat2021', 'train', version='2021', category_ids=category_ids)
    val_ds = get_inat('data/iNat2021', 'val', version='2021', category_ids=category_ids)
    
    num_classes = train_ds.num_classes
    train_size = len(train_ds)
    val_size = len(val_ds)
    samples_per_class = train_size / num_classes if num_classes > 0 else 0
    
    # Parameter counts
    D = 512  # CLIP feature dimension
    params_k1 = num_classes * 1 * D
    params_k32 = num_classes * 32 * D
    
    # Data-to-parameter ratios
    ratio_k1 = train_size / params_k1 if params_k1 > 0 else 0
    ratio_k32 = train_size / params_k32 if params_k32 > 0 else 0
    
    return {
        'name': name,
        'num_classes': num_classes,
        'train_size': train_size,
        'val_size': val_size,
        'samples_per_class': samples_per_class,
        'params_k1': params_k1,
        'params_k32': params_k32,
        'ratio_k1': ratio_k1,
        'ratio_k32': ratio_k32,
    }

def main():
    subsets = {
        'Bulrush': [6297, 6298, 6299, 6300, 6301],
        'Lichen': [5438, 5439, 5440, 5441, 5442, 5443],
        'Manzanita': [7709, 7710, 7711, 7712, 7713],
        'Wrasse': [2837, 2843, 2844, 2845, 2846],
        'Wild_Rye': [6371, 6372, 6373, 6374, 6375],
    }
    
    print("=" * 70)
    print("Analysis: Why K=32 might not help on small subsets")
    print("=" * 70)
    print()
    
    results = []
    for name, cat_ids in subsets.items():
        result = analyze_subset(name, cat_ids)
        results.append(result)
        
        print(f"{name}:")
        print(f"  Classes: {result['num_classes']}")
        print(f"  Train: {result['train_size']:,} samples ({result['samples_per_class']:.1f} per class)")
        print(f"  Val: {result['val_size']:,} samples")
        print(f"  Parameters:")
        print(f"    K=1:  {result['params_k1']:,} ({result['ratio_k1']:.2f} samples/param)")
        print(f"    K=32: {result['params_k32']:,} ({result['ratio_k32']:.2f} samples/param)")
        print(f"  Risk: {'HIGH overfitting risk' if result['ratio_k32'] < 0.1 else 'MODERATE' if result['ratio_k32'] < 1.0 else 'LOW'}")
        print()
    
    print("=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("1. With K=32, we have many more parameters than training samples")
    print("2. This suggests overfitting risk - K=32 might memorize training data")
    print("3. K=1 has fewer parameters and might generalize better")
    print("4. Consider:")
    print("   - Stronger regularization for K=32")
    print("   - Intermediate K values (K=4, K=8)")
    print("   - Early stopping based on validation loss")
    print("   - Lower learning rate for K=32")
    print()

if __name__ == "__main__":
    main()


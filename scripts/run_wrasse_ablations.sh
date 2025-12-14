#!/bin/bash
# Run all Wrasse subset ablations

set -e  # Exit on error

echo "=========================================="
echo "Running Wrasse Subset Ablations"
echo "=========================================="
echo ""

# Create checkpoints directory
mkdir -p checkpoints

# 1. Baseline (Full model K=32)
echo "1/7: Training baseline (Full model K=32)..."
python train.py --config configs/subset_wrasse_baseline.yaml
echo "✓ Baseline complete"
echo ""

# 2. Without semantic initialization
echo "2/7: Training without semantic initialization..."
python train.py --config configs/subset_wrasse_no_semantic_init.yaml
echo "✓ No semantic init complete"
echo ""

# 3. Without mixture loss
echo "3/7: Training without mixture loss..."
python train.py --config configs/subset_wrasse_no_mixture_loss.yaml
echo "✓ No mixture loss complete"
echo ""

# 4. Without regularization
echo "4/7: Training without regularization..."
python train.py --config configs/subset_wrasse_no_regularization.yaml
echo "✓ No regularization complete"
echo ""

# 5. Fixed temperature
echo "5/7: Training with fixed temperature..."
python train.py --config configs/subset_wrasse_fixed_temperature.yaml
echo "✓ Fixed temperature complete"
echo ""

# 6. Hard assignment
echo "6/7: Training with hard assignment..."
python train.py --config configs/subset_wrasse_hard_assignment.yaml
echo "✓ Hard assignment complete"
echo ""

# 7. K=1
echo "7/7: Training with K=1..."
python train.py --config configs/subset_wrasse_k1.yaml
echo "✓ K=1 complete"
echo ""

echo "=========================================="
echo "All ablations complete!"
echo "=========================================="


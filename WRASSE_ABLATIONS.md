# Wrasse Subset Ablation Studies

This document describes the ablation experiments run on the Wrasse subset (5 fish species).

## Ablation Configurations

### 1. Baseline (Full Model K=32)
**Config**: `configs/subset_wrasse_baseline.yaml`
- K=32 sub-prompts per class
- Semantic initialization (offsets initialized near base text features)
- Mixture loss (lambda_mixture=0.5)
- Regularization (offset_reg_weight=0.001)
- Temperature annealing (tau: 1.0 → 0.05)
- Soft assignment (softmax)

### 2. Without Semantic Initialization
**Config**: `configs/subset_wrasse_no_semantic_init.yaml`
- K=32
- **Ablation**: Random initialization (zeros) instead of semantic init
- All other settings same as baseline

### 3. Without Mixture Loss
**Config**: `configs/subset_wrasse_no_mixture_loss.yaml`
- K=32
- **Ablation**: lambda_mixture=0.0 (only classification loss)
- All other settings same as baseline

### 4. Without Regularization
**Config**: `configs/subset_wrasse_no_regularization.yaml`
- K=32
- **Ablation**: offset_reg_weight=0.0 (no L2 regularization on offsets)
- All other settings same as baseline

### 5. Fixed Temperature
**Config**: `configs/subset_wrasse_fixed_temperature.yaml`
- K=32
- **Ablation**: use_tau_annealing=false (tau stays at 1.0, no annealing)
- All other settings same as baseline

### 6. Hard Assignment
**Config**: `configs/subset_wrasse_hard_assignment.yaml`
- K=32
- **Ablation**: use_hard_assignment=true (argmax instead of softmax for gamma)
- All other settings same as baseline

### 7. K=1 (Single Prompt)
**Config**: `configs/subset_wrasse_k1.yaml`
- **Ablation**: K=1 (single prompt per class, no mixture)
- All other settings same as baseline

## Running Ablations

### Run All Ablations Sequentially
```bash
bash scripts/run_wrasse_ablations.sh
```

### Run Individual Ablations
```bash
# Baseline
python train.py --config configs/subset_wrasse_baseline.yaml

# No semantic init
python train.py --config configs/subset_wrasse_no_semantic_init.yaml

# No mixture loss
python train.py --config configs/subset_wrasse_no_mixture_loss.yaml

# No regularization
python train.py --config configs/subset_wrasse_no_regularization.yaml

# Fixed temperature
python train.py --config configs/subset_wrasse_fixed_temperature.yaml

# Hard assignment
python train.py --config configs/subset_wrasse_hard_assignment.yaml

# K=1
python train.py --config configs/subset_wrasse_k1.yaml
```

## Evaluating Ablations

After training, evaluate each model:

```bash
# Baseline
python eval.py --checkpoint checkpoints/best_epoch15.pt --config configs/subset_wrasse_baseline.yaml

# No semantic init
python eval.py --checkpoint checkpoints/best_epoch15.pt --config configs/subset_wrasse_no_semantic_init.yaml

# ... and so on for each ablation
```

## Expected Checkpoint Names

Checkpoints will be saved as:
- `checkpoints/best_epoch*.pt` (best validation accuracy)
- `checkpoints/checkpoint_epoch*.pt` (latest epoch)

You may want to rename them for clarity:
```bash
mv checkpoints/best_epoch15.pt checkpoints/wrasse_baseline.pt
mv checkpoints/best_epoch15.pt checkpoints/wrasse_no_semantic_init.pt
# etc.
```

## Model Changes

The following model parameters were added to support ablations:
- `use_semantic_init`: If False, initializes offsets to zero instead of random near base features
- `offset_reg_weight`: L2 regularization strength (0.0 = no regularization)
- `use_hard_assignment`: If True, uses argmax instead of softmax for assignment

The following training parameters control ablations:
- `lambda_mixture`: Weight for mixture loss (0.0 = no mixture loss)
- `use_tau_annealing`: If False, temperature stays fixed at em_tau_start


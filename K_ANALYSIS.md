# Analysis: Why K=32 Doesn't Help on Small Subsets

## The Problem

Your observation is correct: **K=32 doesn't help (and sometimes hurts) on small iNaturalist subsets**. Here's why:

## Dataset vs Parameter Analysis

| Subset | Classes | Train Samples | K=1 Params | K=32 Params | Samples/Param (K=32) |
|--------|---------|---------------|------------|-------------|---------------------|
| Bulrush | 5 | 1,350 | 2,560 | 81,920 | **0.016** |
| Lichen | 6 | 1,632 | 3,072 | 98,304 | **0.017** |
| Manzanita | 5 | 1,423 | 2,560 | 81,920 | **0.017** |
| Wrasse | 5 | 1,232 | 2,560 | 81,920 | **0.015** |

## Key Findings

1. **Severe Overparameterization**: With K=32, you have **~82k-98k parameters** but only **~1,200-1,600 training samples**
   - That's only **0.015-0.017 samples per parameter**!
   - K=1 has **0.4-0.6 samples per parameter** (much healthier)

2. **Overfitting Risk**: K=32 models likely memorize training data rather than learning generalizable patterns

3. **Why K=1 Works Better**: 
   - Fewer parameters (2.5k-3k) relative to data
   - Less prone to overfitting
   - Still benefits from learnable prompt offsets

## Solutions to Try

### Option 1: Stronger Regularization for K=32
Increase regularization to prevent overfitting:
```yaml
model:
  offset_reg_weight: 0.01  # 10x stronger (default: 0.001)
```

### Option 2: Intermediate K Values
Try K=4 or K=8 as a middle ground:
- K=4: ~10k-12k parameters (0.1-0.15 samples/param)
- K=8: ~20k-24k parameters (0.05-0.08 samples/param)

### Option 3: Lower Learning Rate for K=32
```yaml
train:
  lr: 1e-4  # Lower LR to prevent overfitting
```

### Option 4: Early Stopping
Monitor validation loss and stop early if it starts increasing.

### Option 5: Accept K=1 for Small Subsets
For subsets with <2,000 training samples, K=1 may be optimal.

## Recommended Experiments

1. **K=4 with stronger regularization**:
   ```yaml
   model:
     K: 4
     offset_reg_weight: 0.01
   ```

2. **K=32 with much stronger regularization**:
   ```yaml
   model:
     K: 32
     offset_reg_weight: 0.01  # or even 0.1
   ```

3. **K=32 with lower learning rate**:
   ```yaml
   model:
     K: 32
   train:
     lr: 1e-4  # instead of 5e-4
   ```

## Conclusion

The mixture-of-prompts approach (K>1) is designed for **large-scale fine-grained datasets** where there's enough data to learn meaningful specializations. For small subsets (5-6 classes, ~1,500 samples), **K=1 is likely optimal** because:

- It has enough capacity to learn class-specific prompts
- It doesn't overfit
- It generalizes better to validation set

This is actually a **feature, not a bug** - the model adapts its complexity to the dataset size!


# Mode Collapse Analysis

## Problem Identified

The model is experiencing **mode collapse** - all K=8 sub-prompts for each class have converged to nearly identical representations.

### Evidence

1. **Final prompt features are nearly identical**:
   - Pairwise cosine similarity between sub-prompts: **0.94-0.96** (should be much lower for diversity)
   - Mean similarity: **0.94** (nearly identical vectors)

2. **Visualization shows one dominant sub-prompt**:
   - One sub-prompt (k=3 for class 0) gets ~99% of assignments (gamma ~0.99)
   - Other sub-prompts get <1% (gamma ~0.001-0.01)
   - Images look similar across sub-prompts (repeated groups)

3. **Prompt offsets are diverse, but final features are not**:
   - Raw offsets have cosine similarity of ~0.15 (diverse)
   - But after adding to base features and normalizing, they become identical
   - This suggests offsets are too small relative to base features

## Root Causes

1. **No Diversity Loss**: The training objective has no mechanism to encourage sub-prompts to be different from each other
   - Mixture loss only maximizes similarity between images and their assigned sub-prompt
   - Classification loss uses max pooling (only cares if one sub-prompt is good)
   - Regularization penalizes large offsets, keeping them small

2. **Temperature Annealing Too Aggressive**:
   - Temperature drops from 1.0 → 0.05
   - At 0.05, softmax becomes extremely peaked (hard assignments)
   - Once one sub-prompt gets all assignments, all others collapse to match it

3. **Offset Regularization Too Strong**:
   - `offset_reg_weight=0.001` keeps offsets small
   - Small offsets + normalization = all prompts become similar
   - Base features dominate, offsets become negligible after normalization

4. **No Entropy Regularization**:
   - No penalty for having uniform assignments vs. concentrated assignments
   - Model can collapse all assignments to one sub-prompt without penalty

## Solutions

### Option 1: Add Diversity Loss (Recommended)
Add a loss term that encourages sub-prompts within a class to be different:

```python
# In mop_clip.py forward() method, after computing prompt_feats:
# Diversity loss: penalize high similarity between sub-prompts
diversity_loss = 0.0
for i in range(self.K):
    for j in range(i+1, self.K):
        # Compute cosine similarity between sub-prompts
        sim = F.cosine_similarity(
            prompt_feats[:, i:i+1], 
            prompt_feats[:, j:j+1], 
            dim=2
        ).mean()
        # Penalize high similarity (we want them to be different)
        diversity_loss += sim ** 2

diversity_weight = 0.1  # Tune this
total_loss = loss + diversity_weight * diversity_loss
```

### Option 2: Higher Final Temperature
Keep temperature higher to maintain soft assignments:

```yaml
model:
  em_tau_end: 0.3  # Instead of 0.05
```

### Option 3: Entropy Regularization
Penalize concentrated assignments, encourage spreading:

```python
# In forward(), after computing gamma:
entropy = -(gamma * torch.log(gamma + 1e-10)).sum(dim=1).mean()
entropy_loss = -entropy  # Negative because we want to maximize entropy
# Add to total loss
```

### Option 4: Weaker Offset Regularization + Larger Offsets
Allow offsets to be larger so they can create diversity:

```yaml
model:
  offset_reg_weight: 0.0001  # 10x weaker (allow larger offsets)
```

### Option 5: Orthogonal Initialization
Initialize sub-prompts to be orthogonal to encourage diversity from the start:

```python
# In initialization, use orthogonal initialization for offsets
torch.nn.init.orthogonal_(self.prompt_offsets)
```

### Option 6: Minimum Distance Constraint
Force sub-prompts to maintain minimum distance:

```python
# Add loss that encourages sub-prompts to be at least d_min apart
d_min = 0.5  # Minimum cosine distance
for i in range(self.K):
    for j in range(i+1, self.K):
        sim = F.cosine_similarity(...)
        # Penalize if too similar
        if sim > (1.0 - d_min):
            diversity_loss += (sim - (1.0 - d_min)) ** 2
```

## Recommended Fix

**Combine Options 1 + 2 + 3**:
1. Add diversity loss to prevent collapse
2. Use higher final temperature (0.3 instead of 0.05)
3. Add entropy regularization to encourage spreading

This will encourage the model to learn distinct sub-prompts that capture different modes/variations within each class.


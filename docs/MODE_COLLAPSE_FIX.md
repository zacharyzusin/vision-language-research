# Mode Collapse Fix Implementation

## Changes Made

### 1. **Diversity Loss** (`diversity_loss_weight: 0.1`)
Prevents sub-prompts from collapsing to the same representation by penalizing high cosine similarity between sub-prompts within the same class.

**Implementation**:
- Computes pairwise cosine similarities between all sub-prompts for classes in the batch
- Penalizes high similarity values (squared) to encourage diversity
- Formula: `diversity_loss = mean(pairwise_similarities²)`

**Why it works**:
- Directly addresses the root cause: sub-prompts becoming too similar
- Encourages the model to learn distinct representations
- Especially important for smaller K values (K=8)

### 2. **Entropy Regularization** (`entropy_loss_weight: 0.01`)
Encourages spreading of assignments across sub-prompts rather than collapsing to a single dominant sub-prompt.

**Implementation**:
- Computes entropy of assignment distribution: `entropy = -sum(gamma * log(gamma))`
- Maximizes entropy (minimizes negative entropy) to encourage uniform spreading
- Only active when using soft assignments (not hard assignments)

**Why it works**:
- Prevents one sub-prompt from getting all assignments (gamma ~1.0, others ~0.0)
- Encourages more balanced distributions across sub-prompts
- Works synergistically with diversity loss

### 3. **Higher Final Temperature** (`em_tau_end: 0.3`)
Increased from 0.05 to 0.3 to maintain softer assignments throughout training.

**Why it works**:
- Lower temperature (0.05) causes extremely peaked softmax → hard assignments
- Higher temperature (0.3) maintains soft assignments → allows diversity to be learned
- Still allows specialization (not as soft as 1.0) but prevents complete collapse

## Configuration Changes

### `configs/stanford_cars.yaml` and `configs/default.yaml`
```yaml
model:
  em_tau_end: 0.3  # Increased from 0.05
  diversity_loss_weight: 0.1  # New parameter
  entropy_loss_weight: 0.01   # New parameter
```

## Code Changes

### `src/models/mop_clip.py`
1. Added `diversity_loss_weight` and `entropy_loss_weight` to `__init__`
2. Implemented diversity loss computation in `forward()`
3. Implemented entropy regularization in `forward()`
4. Added new loss components to return dictionary

### `train.py`
1. Added loading of new hyperparameters from config
2. Updated model initialization to pass new parameters
3. Updated progress bar to display diversity and entropy losses

## Expected Effects

### Before (Mode Collapse):
- Sub-prompts have cosine similarity ~0.94-0.96 (nearly identical)
- One sub-prompt gets ~99% of assignments (gamma ~0.99)
- Other sub-prompts get <1% (gamma ~0.001-0.01)
- Visualizations show repeated images across sub-prompts

### After (With Fixes):
- Sub-prompts have lower cosine similarity (more diverse)
- Assignments spread more evenly across sub-prompts
- Each sub-prompt learns distinct visual modes/variations
- Visualizations show distinct images for each sub-prompt

## Tuning the Hyperparameters

If mode collapse still occurs, you can adjust:

1. **Increase diversity loss weight**: `diversity_loss_weight: 0.2` (stronger penalty)
2. **Increase entropy weight**: `entropy_loss_weight: 0.02` (stronger spreading)
3. **Increase final temperature**: `em_tau_end: 0.5` (softer assignments)

If sub-prompts become too diverse (losing class coherence):
1. **Decrease diversity loss weight**: `diversity_loss_weight: 0.05`
2. **Decrease entropy weight**: `entropy_loss_weight: 0.005`

## Testing

To verify the fixes work:
1. Train a model with the new hyperparameters
2. Generate visualizations for multiple classes
3. Check that:
   - Sub-prompts show distinct images (not repeats)
   - Gamma values are spread more evenly (not just one ~1.0)
   - Pairwise cosine similarities between sub-prompts are <0.9

## Backward Compatibility

The new parameters have default values, so existing configs without these parameters will still work:
- `diversity_loss_weight: 0.1` (default)
- `entropy_loss_weight: 0.01` (default)
- If not specified, losses are simply not applied


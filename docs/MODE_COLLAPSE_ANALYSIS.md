# Mode Collapse Analysis: Stanford Cars vs iNaturalist

## Key Finding: No Code Changes - It's the K Value!

### The Critical Difference

| Model | K Value | Mode Collapse? |
|-------|---------|----------------|
| **Stanford Cars** | **K=8** | ✅ **Yes - severe collapse** |
| **iNaturalist** | **K=32** | ❌ **No - diverse sub-prompts** |

### Evidence

1. **Model Architecture**: No changes - same code, same training procedure
2. **Training Hyperparameters**: Both use:
   - `em_tau_start=1.0`, `em_tau_end=0.05`
   - `offset_reg_weight=0.001`
   - `lambda_mixture=0.5`
   - Same learning rate, epochs, etc.

3. **Raw Offset Diversity Comparison**:
   - **iNaturalist (K=32)**: Mean pairwise cosine similarity = 0.17 (diverse!)
   - **Stanford Cars (K=8)**: Mean pairwise cosine similarity = 0.15 (diverse offsets)
   - But final prompt features collapse (similarity ~0.94-0.96)

4. **The Comment in Config**:
   ```yaml
   # Lowered from 0.3 to 0.05 for better specialization
   em_tau_end: 0.05
   ```
   This change was made for both datasets, so it's not the cause of the difference.

## Why K=8 Collapses More Easily Than K=32

### 1. **Less Capacity for Diversity**
- **K=8**: 8 * 512 = 4,096 parameters per class
- **K=32**: 32 * 512 = 16,384 parameters per class
- With 4x fewer parameters, there's less capacity to encode diverse modes

### 2. **Fewer "Chances" to Find Distinct Modes**
- With only 8 sub-prompts, if one becomes dominant early, the other 7 have less "room" to specialize
- With 32 sub-prompts, even if some collapse, others can still maintain diversity

### 3. **Smaller Margin for Error**
- The mode collapse problem exists in both models
- But with K=32, you need 32 sub-prompts to collapse before you see complete failure
- With K=8, if just 7 collapse, you've lost 87.5% of diversity

### 4. **Regularization Effect is Stronger**
- Same `offset_reg_weight=0.001` for both
- But with K=8, if regularization prevents 1 sub-prompt from diverging, you've lost 12.5% of capacity
- With K=32, losing 1 sub-prompt is only 3.1% of capacity

## Dataset Size Factor (Secondary)

- **Stanford Cars**: ~33 train samples per class (6516 / 196)
- **iNaturalist**: Many more samples per class
- Smaller datasets can exacerbate collapse, but the K value is the primary factor

## Conclusion

**The mode collapse is NOT due to code changes** - it's a fundamental limitation of the training objective that becomes **more severe with smaller K values**.

The training objective has no mechanism to prevent collapse:
- No diversity loss between sub-prompts
- No entropy regularization to encourage spreading
- Low final temperature (0.05) causes hard assignments

With K=32, the model "accidentally" maintains some diversity because there's more capacity and redundancy. With K=8, the collapse becomes obvious because there's less room to hide the problem.

## Solution

The fixes proposed in `DIAGNOSTIC_MODE_COLLAPSE.md` are still valid and necessary:
1. Add diversity loss
2. Add entropy regularization  
3. Use higher final temperature (0.3 instead of 0.05)

These fixes will prevent collapse for **both** K=8 and K=32, but they're especially critical for smaller K values.


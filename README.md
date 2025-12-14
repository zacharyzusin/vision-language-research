# Mixture-of-Prompts CLIP for Fine-Grained Classification

This repository implements a Mixture-of-Prompts (MoP) approach for fine-grained visual classification using CLIP. The method extends CLIP with hierarchical text prompts and learnable per-class prompt offsets to enable better specialization for fine-grained tasks like species classification.

## Overview

The model uses:
- **Hierarchical text prompts**: Combines species, genus, family, and order information
- **Learnable sub-prompts**: Each class has K learnable prompt offsets that specialize for different visual patterns
- **Soft EM-like training**: Images are softly assigned to sub-prompts during training
- **Dual loss objective**: Combines mixture loss (intra-class specialization) and classification loss (inter-class separation)

## Project Structure

```
iNatCode/
├── src/
│   ├── models/
│   │   └── mop_clip.py          # MixturePromptCLIP model implementation
│   └── datasets/
│       └── inat_dataset.py      # iNaturalist dataset loading utilities (supports 2018 & 2021)
├── scripts/
│   ├── convert_inat2018.py      # Dataset format conversion script (supports 2018 & 2021)
│   └── visualize_subprompts.py  # Visualization of sub-prompt assignments
├── configs/
│   └── default.yaml             # Training configuration
├── train.py                      # Main training script
├── eval.py                       # Evaluation script
├── test_model.py                 # Comprehensive model test suite
└── clip_zeroshot_baseline.py     # Zero-shot CLIP baseline for comparison
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- CLIP (OpenAI): `pip install git+https://github.com/openai/CLIP.git`
- Other dependencies: `pip install wandb tqdm pyyaml`

### Dataset Setup

1. Download iNaturalist 2021 dataset (or 2018 if preferred)
2. Ensure the following structure:
   ```
   data/iNat2021/
   ├── 2021/                      # Image directories
   ├── train2021.json
   ├── val2021.json
   └── categories.json
   ```
   
   For iNaturalist 2018, use:
   ```
   data/iNat2018/
   ├── 2018/                      # Image directories
   ├── train2018.json
   ├── val2018.json
   └── categories.json
   ```

3. (Optional) Convert to torchvision format using:
   ```bash
   python scripts/convert_inat2018.py --data_root data/iNat2021 --version 2021
   ```
   
   For 2018:
   ```bash
   python scripts/convert_inat2018.py --data_root data/iNat2018 --version 2018
   ```

## Usage

### Training

1. Configure training parameters in `configs/default.yaml`:
   ```yaml
   dataset:
     root: data/iNat2021
     version: 2021              # Use "2018" for iNaturalist 2018
   
   model:
     clip_model: ViT-B/16
     K: 32                    # Number of sub-prompts per class
     em_tau_start: 1.0        # Initial temperature
     em_tau_end: 0.05         # Final temperature
   
   train:
     batch_size: 16
     lr: 5e-4
     epochs: 30
     warmup_steps: 2000
     lambda_mixture: 0.5      # Weight for mixture vs classification loss
     temp_cls: 0.07          # Temperature for classification logits
   ```

2. Run training:
   ```bash
   python train.py --config configs/default.yaml
   ```

3. Resume from checkpoint:
   ```bash
   python train.py --config configs/default.yaml --resume checkpoints/best_epoch15.pt
   ```

### Evaluation

Evaluate a trained model:
```bash
python eval.py --checkpoint checkpoints/best_epoch15.pt --data_root data/iNat2021 --version 2021 --batch_size 32
```

For iNaturalist 2018:
```bash
python eval.py --checkpoint checkpoints/best_epoch15.pt --data_root data/iNat2018 --version 2018 --batch_size 32
```

### Testing

Before training, verify model functionality:
```bash
python test_model.py
```

This runs comprehensive tests including:
- Model initialization
- Forward pass and loss computation
- Gradient flow
- Prediction
- Temperature annealing effects
- Training step simulation

### Visualization

Visualize which images are assigned to each sub-prompt for a given class:
```bash
python scripts/visualize_subprompts.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_epoch15.pt \
    --class_name "danaus plexippus" \
    --split val \
    --top_n 16
```

Options:
- `--class_idx`: Specify class by index (0-based)
- `--class_name`: Specify class by species name
- `--split`: Dataset split (train/val)
- `--top_n`: Number of top images per sub-prompt to visualize
- `--out_dir`: Output directory for visualization grids

### Baseline Comparison

Run zero-shot CLIP baseline:
```bash
python clip_zeroshot_baseline.py --data_root data/iNat2021 --version 2021 --clip_model ViT-B/16 --batch_size 64
```

For iNaturalist 2018:
```bash
python clip_zeroshot_baseline.py --data_root data/iNat2018 --version 2018 --clip_model ViT-B/16 --batch_size 64
```

## Model Architecture

### Key Components

1. **Base Text Features**: Hierarchical prompts encoded with CLIP text encoder
   - Templates: "a photo of {species}", "a wildlife photo of the species {scientific_name}", etc.
   - Averaged across templates and normalized

2. **Learnable Prompt Offsets**: Per-class, per-sub-prompt learnable offsets
   - Shape: `(num_classes, K, embedding_dim)`
   - Initialized with small random values

3. **Final Prompts**: `normalize(base_text_features[c] + prompt_offsets[c, k])`

### Training Objective

The model optimizes a combined loss:
```
L = λ * L_mixture + (1-λ) * L_cls + L_reg
```

- **L_mixture**: Soft EM-like loss encouraging sub-prompts to specialize
- **L_cls**: Classification loss using max-pooling over sub-prompts
- **L_reg**: L2 regularization on prompt offsets

### Temperature Annealing

The temperature parameter `em_tau` is annealed during training:
- Starts at `em_tau_start` (softer assignments)
- Ends at `em_tau_end` (sharper, more specialized assignments)
- Enables gradual specialization of sub-prompts

## Configuration

Key hyperparameters in `configs/default.yaml`:

- **K**: Number of sub-prompts per class (default: 32)
- **em_tau_start/end**: Temperature schedule for soft assignments
- **lambda_mixture**: Balance between mixture and classification losses
- **temp_cls**: Temperature scaling for classification logits
- **lr**: Learning rate (default: 5e-4)
- **warmup_steps**: Linear warmup steps before cosine decay

## Checkpoints

Checkpoints are saved in `checkpoints/` directory and include:
- Model state dict
- Optimizer state dict
- Current epoch and step
- Best validation accuracy
- Current temperature (tau)

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Mixture-of-Prompts CLIP for Fine-Grained Classification},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

- OpenAI CLIP: https://github.com/openai/CLIP
- iNaturalist dataset: https://github.com/visipedia/inat_comp

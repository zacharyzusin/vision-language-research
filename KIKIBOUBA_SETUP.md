# KikiBouba Dataset Setup and Training

This guide explains how to use the KikiBouba dataset with the MoP-CLIP training pipeline.

## Dataset Overview

KikiBouba is a multiclass classification dataset with five classes:
- **bouba**: Round, curved shapes (label 0)
- **galaga**: Sharp, angular shapes variant (label 1)
- **kepike**: Sharp, angular shapes variant (label 2)
- **kiki**: Sharp, angular shapes variant (label 3)
- **maluma**: Sharp, angular shapes variant (label 4)

The dataset structure:
```
kiki_bouba_v2_split/
    train/
        bouba/      (844 images)
        galaga/     (829 images)
        kepike/     (855 images)
        kiki/       (812 images)
        maluma/     (837 images)
    val/
        bouba/      (similar structure)
        galaga/
        kepike/
        kiki/
        maluma/
```

Each directory is treated as a separate class for multiclass classification.

## Dataset Statistics

- **Training samples**: 4,177 total
  - bouba: 844 images
  - galaga: 829 images
  - kepike: 855 images
  - kiki: 812 images
  - maluma: 837 images
- **Validation samples**: 1,023
- **Classes**: 5 (multiclass classification)

## Quick Start

### 1. Download Dataset

The dataset has been downloaded to: `data/kikibouba/kiki_bouba_v2_split/`

If you need to download it again:
```bash
cd data/kikibouba
gdown "https://drive.google.com/uc?id=17ibF3tzFiZrMb9ZnpYlLEh-xmWkPJpNH" -O kiki_bouba_v2_split.zip
unzip kiki_bouba_v2_split.zip
```

### 2. Train Model

```bash
python train.py --config configs/kikibouba.yaml
```

### 3. Evaluate Model

```bash
python eval.py --checkpoint checkpoints/best_epoch15.pt \
    --data_root data/kikibouba \
    --version kikibouba
```

## Configuration

The config file `configs/kikibouba.yaml` includes:
- Dataset type: `kikibouba` (uses custom loader)
- Batch size: 32 (can be larger for binary classification)
- Model: ViT-B/16 with K=32 sub-prompts
- Optimized for smaller dataset (fewer warmup steps)

## How It Works

1. **Dataset Loading**: `KikiBoubaDataset` automatically:
   - Finds all class directories (bouba, galaga, kepike, kiki, maluma)
   - Maps each directory to a unique label (0-4)
   - Loads images with CLIP preprocessing

2. **Metadata**: Creates hierarchical metadata compatible with MoP-CLIP:
   - Each class directory becomes its own class
   - Classes are sorted alphabetically for consistent ordering

3. **Training**: Uses the same MoP-CLIP training pipeline as iNaturalist, adapted for 5-class classification.

## Notes

- The dataset treats each directory as a separate class (multiclass classification)
- This is a 5-class classification task (bouba, galaga, kepike, kiki, maluma)
- Training should be much faster than iNaturalist (4K vs 2.7M samples)
- The model architecture adapts automatically to 5 classes


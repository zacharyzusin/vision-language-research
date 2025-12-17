# KikiBouba Dataset Setup and Training

This guide explains how to use the KikiBouba dataset with the MoP-CLIP training pipeline.

## Dataset Overview

We support **two related KikiBouba datasets**:

- **KikiBouba v2** (renamed/cleaned version; class names: bouba/galaga/kepike/kiki/maluma)
- **KikiBouba v1** (original release; class names: bamaba/duludu/gaduga/lomulo/nomano)

Both are treated as 5-way multiclass classification problems.

### KikiBouba v2

KikiBouba v2 has five classes:
- **bouba**
- **galaga**
- **kepike**
- **kiki**
- **maluma**

The v2 dataset structure:
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

### KikiBouba v1

We also support the original **KikiBouba v1** release. It has five classes:
- **bamaba**
- **duludu**
- **gaduga**
- **lomulo**
- **nomano**

Its folder layout is compatible as long as it ends up in one of these forms
under `data/kikibouba/`:

```
data/kikibouba/
    kiki_bouba_v1_split/
        train/
            bouba/
            galaga/
            ...
        val/   # or test/
            bouba/
            galaga/
            ...
```

or

```
data/kikibouba/
    train/
        bouba/
        galaga/
        ...
    val/   # or test/
        bouba/
        galaga/
        ...
```

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

If you need to download **v2** again:
```bash
cd data/kikibouba
gdown "https://drive.google.com/uc?id=17ibF3tzFiZrMb9ZnpYlLEh-xmWkPJpNH" -O kiki_bouba_v2_split.zip
unzip kiki_bouba_v2_split.zip
```

To use **KikiBouba v1** (for example, from
`https://drive.google.com/file/d/1s26MlkNJUXTvuthj5Q2g2WRIx5ttP1Ei/view`):

1. Download the file (via browser or `gdown`) into `data/kikibouba/`.
2. Unzip/extract it so that you end up with either:
   - `data/kikibouba/kiki_bouba_v1_split/train/...` and `val/` (or `test/`), or
   - `data/kikibouba/train/...` and `val/` (or `test/`) directly.

The loader will automatically detect both v1 and v2 layouts; you do **not**
need to change any code or config as long as the split folders are in one of
those locations.

### 2. Train Model (separate configs for v1 and v2)

- **Train on KikiBouba v2**:

```bash
python train.py --config configs/kikibouba_v2.yaml
```

- **Train on KikiBouba v1**:

```bash
python train.py --config configs/kikibouba_v1.yaml
```

Both configs point to `data/kikibouba` but set `dataset.version` differently so
you can run completely separate trainings.

### 3. Evaluate Model

For example, to evaluate a trained checkpoint:

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


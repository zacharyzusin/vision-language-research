# Fine-Grained Subset Training Guide

This guide explains how to train and evaluate models on fine-grained subsets of iNaturalist 2021.

## Available Subsets

The codebase supports training on the following taxonomic subsets:

1. **Lichen** (5 species): Teloschistes and Xanthoria species
2. **Wrasse** (5 species): Halichoeres and Thalassoma fish species
3. **Wild Rye** (5 species): Elymus grass species
4. **Manzanita** (5 species): Arctostaphylos berry shrub species
5. **Bulrush** (5 species): Scirpus herb species

## Quick Start

### Training on a Subset

Use the pre-configured subset config files:

```bash
# Train on Lichen subset
python train.py --config configs/subset_lichen.yaml

# Train on Wrasse subset
python train.py --config configs/subset_wrasse.yaml

# Train on Wild Rye subset
python train.py --config configs/subset_wild_rye.yaml

# Train on Manzanita subset
python train.py --config configs/subset_manzanita.yaml

# Train on Bulrush subset
python train.py --config configs/subset_bulrush.yaml
```

### Evaluating on a Subset

```bash
# Evaluate using config file
python eval.py --checkpoint checkpoints/best_epoch15.pt --config configs/subset_lichen.yaml

# Or specify category_ids directly
python eval.py --checkpoint checkpoints/best_epoch15.pt \
    --category_ids 5439 5440 5441 5442 5443 \
    --data_root data/iNat2021 --version 2021
```

## Subset Details

### Lichen (5 species)
- Category IDs: [5439, 5440, 5441, 5442, 5443]
- Species:
  - Teloschistes chrysophthalmus
  - Teloschistes exilis
  - Teloschistes flavicans
  - Xanthomendoza fallax
  - Xanthoria parietina

### Wrasse (5 species)
- Category IDs: [2837, 2843, 2844, 2845, 2846]
- Species:
  - Halichoeres bivittatus
  - Thalassoma bifasciatum
  - Thalassoma hardwicke
  - Thalassoma lucasanum
  - Thalassoma lunare

### Wild Rye (5 species)
- Category IDs: [6371, 6372, 6373, 6374, 6375]
- Species:
  - Elymus canadensis
  - Elymus elymoides
  - Elymus hystrix
  - Elymus repens
  - Elymus virginicus

### Manzanita (5 species)
- Category IDs: [7709, 7710, 7711, 7712, 7713]
- Species:
  - Arctostaphylos glauca
  - Arctostaphylos nevadensis
  - Arctostaphylos patula
  - Arctostaphylos pungens
  - Arctostaphylos uva-ursi

### Bulrush (5 species)
- Category IDs: [6297, 6298, 6299, 6300, 6301]
- Species:
  - Scirpus atrovirens
  - Scirpus cyperinus
  - Scirpus microcarpus
  - Scirpus pendulus
  - Scirpus sylvaticus

## Creating Custom Subsets

To create a custom subset:

1. **Find category IDs** for your species:
   ```python
   import json
   with open('data/iNat2021/categories.json', 'r') as f:
       categories = json.load(f)
   
   # Search for species
   for cat in categories:
       if 'your_species_name' in cat['name'].lower():
           print(f"{cat['id']}: {cat['name']}")
   ```

2. **Create a config file** (e.g., `configs/subset_custom.yaml`):
   ```yaml
   dataset:
     root: data/iNat2021
     version: 2021
     category_ids: [id1, id2, id3, ...]  # Your category IDs
   
   model:
     clip_model: ViT-B/16
     K: 32
     em_tau_start: 1.0
     em_tau_end: 0.05
   
   train:
     batch_size: 16
     lr: 5e-4
     epochs: 30
     warmup_steps: 2000
     lambda_mixture: 0.5
     temp_cls: 0.07
   ```

3. **Train**:
   ```bash
   python train.py --config configs/subset_custom.yaml
   ```

## How It Works

1. **Dataset Filtering**: The `get_inat()` function accepts a `category_ids` parameter that filters the dataset to only include samples from those categories.

2. **Label Remapping**: Labels are automatically remapped from the original category IDs to [0, num_subset_classes-1] for proper training.

3. **Metadata Filtering**: The `extract_hierarchical_metadata()` function also filters metadata to only include the subset species, ensuring prompt generation works correctly.

4. **Model Initialization**: The model is initialized with the correct number of classes for the subset, and prompts are generated only for the subset species.

## Notes

- Each subset trains independently - you train separate models for each subset
- Labels are automatically remapped to [0, num_classes-1] for each subset
- The model architecture adapts to the number of classes in the subset
- All training features (checkpointing, wandb logging, etc.) work with subsets

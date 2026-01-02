# iNaturalist 2021 Dataset Download Guide

This guide explains how to download and set up the iNaturalist 2021 dataset for use with this codebase.

## Official Sources

- **Competition Repository**: https://github.com/visipedia/inat_comp/tree/master/2021
- **Dataset Size**: ~318 GB total
  - Training images: ~270 GB (2.7M images)
  - Validation images: ~10 GB (100K images)
  - Test images: ~38 GB (500K images)

## Quick Start

### Option 1: Automated Download Script

```bash
# Make script executable
chmod +x scripts/download_inat2021.sh

# Run download script (will prompt for large image downloads)
./scripts/download_inat2021.sh data/iNat2021
```

### Option 2: Manual Download

#### Step 1: Download Annotation Files

These are small files (~100MB total) and required for training:

```bash
mkdir -p data/iNat2021
cd data/iNat2021

# Download training annotations
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz
tar -xzf train.json.tar.gz
rm train.json.tar.gz

# Download validation annotations
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz
tar -xzf val.json.tar.gz
rm val.json.tar.gz
```

#### Step 2: Download Images (Optional but Required for Training)

**Warning**: These files are very large (~300GB total). Ensure you have:
- Sufficient disk space
- Stable internet connection
- Time (downloads may take hours)

```bash
# Training images (~270GB, 2.7M images)
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
tar -xzf train.tar.gz
# Optionally remove archive: rm train.tar.gz

# Validation images (~10GB, 100K images)
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
tar -xzf val.tar.gz
# Optionally remove archive: rm val.tar.gz
```

#### Step 3: Organize Directory Structure

After extraction, your directory should look like:

```
data/iNat2021/
├── train2021.json          # Training annotations
├── val2021.json            # Validation annotations
├── categories.json         # Category metadata (may need to extract)
└── 2021/                   # Image directories
    ├── Aves/
    │   └── 2761/
    │       └── *.jpg
    └── ...
```

**Note**: The extracted image directory might be named `train/` or `val/` instead of `2021/`. If so, you may need to:
- Rename it to `2021/`, or
- Update the code to look in the correct location

#### Step 4: Extract categories.json

The `categories.json` file may be included in the annotation archives. If not present, you can extract category information from `train2021.json`:

```python
import json

# Load training annotations
with open('data/iNat2021/train2021.json', 'r') as f:
    data = json.load(f)

# Extract unique categories
categories = {}
for ann in data['annotations']:
    cat_id = ann['category_id']
    if cat_id not in categories:
        # Find category info in data
        # (structure may vary, adjust as needed)
        pass

# Save categories.json
with open('data/iNat2021/categories.json', 'w') as f:
    json.dump(list(categories.values()), f, indent=2)
```

Alternatively, `categories.json` might be available separately or embedded in the annotation files.

## Verify Dataset Setup

After downloading, verify your setup:

```bash
python scripts/check_dataset.py --root data/iNat2021 --version 2021
```

This will check:
- ✓ All required JSON files exist
- ✓ Image directory exists
- ✓ Directory structure is correct

## Dataset Statistics

- **Total Species**: 10,000
- **Training Images**: ~2.7 million
- **Validation Images**: 100,000
- **Test Images**: 500,000
- **Mini Training Split**: 500,000 images (50 per species) - also available

## Alternative: Mini Training Split

If you want a smaller dataset for testing, you can download the "mini" training split:

```bash
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
tar -xzf train_mini.tar.gz
```

This contains 50 images per species (500,000 total images).

## MD5 Checksums (for verification)

You can verify download integrity using MD5 checksums:

```bash
# Training images
md5sum train.tar.gz
# Expected: e0526d53c7f7b2e3167b2b43bb2690ed

# Check other files as provided in the repository
```

## Troubleshooting

### Issue: "No JSON files found"
- Ensure you've downloaded and extracted `train2021.json` and `val2021.json`
- Check that files are in `data/iNat2021/` directory

### Issue: "Image directory not found"
- Check that images were extracted
- Verify the directory name (might be `train/` or `val/` instead of `2021/`)
- Update the code or rename directories as needed

### Issue: "File paths don't match"
- The JSON `file_name` field should match actual file locations
- You may need to adjust paths in the code or reorganize files

## References

- Official Competition: https://www.kaggle.com/c/inaturalist-2021
- GitHub Repository: https://github.com/visipedia/inat_comp/tree/master/2021
- TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/i_naturalist2021

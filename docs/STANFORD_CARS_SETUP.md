# Stanford Cars Dataset Setup Guide

The Stanford Cars dataset contains 196 classes of cars with 8,144 training images and 8,041 test images.

## Quick Start: Automated Download

Use the provided download script that tries multiple alternative sources:

```bash
# Python script (recommended - tries GitHub, Hugging Face, etc.)
python scripts/download_stanford_cars.py [data/stanford_cars]

# Or bash script (tries wget/curl from various URLs)
bash scripts/download_stanford_cars.sh [data/stanford_cars]
```

The script will automatically:
- Try downloading from GitHub repositories
- Try downloading from Hugging Face (if datasets library is installed)
- Extract and organize files correctly
- Verify the dataset structure

**Requirements for Python script:**
```bash
pip install requests tqdm
# Optional for Hugging Face:
pip install datasets
```

## Download Instructions

### Option 1: Automated Download Script (Recommended)

1. Visit the official Stanford Cars dataset page:
   https://ai.stanford.edu/~jkrause/cars/car_dataset.html

2. Download the following files:
   - `car_ims.tgz` - Contains all images (~1.5 GB)
   - `cars_annos.mat` - Annotations file
   - `cars_meta.mat` - Metadata file (optional, but recommended)

3. Extract and organize the files:

```bash
mkdir -p data/stanford_cars
cd data/stanford_cars

# Extract images
tar -xzf car_ims.tgz

# Move annotation files to the root directory
# (if they're in a subdirectory after extraction)
mv cars_annos.mat .
mv cars_meta.mat .  # if available
```

### Option 2: Use Alternative Download Script

If the official link is broken, you can try downloading from alternative sources:

```bash
# Create directory
mkdir -p data/stanford_cars
cd data/stanford_cars

# Try downloading from alternative mirrors (if available)
# Note: You may need to find a working mirror
wget <alternative_url>/car_ims.tgz
wget <alternative_url>/cars_annos.mat
wget <alternative_url>/cars_meta.mat

# Extract
tar -xzf car_ims.tgz
```

## Expected Directory Structure

After setup, your directory should look like:

```
data/stanford_cars/
├── car_ims/              # All images
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── cars_annos.mat        # Annotations (required)
└── cars_meta.mat         # Metadata (optional but recommended)
```

## Alternative: Directory-Based Structure

If you prefer a class-based directory structure, you can organize it as:

```
data/stanford_cars/
├── train/
│   ├── class_001/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class_002/
│   └── ...
└── test/
    ├── class_001/
    ├── class_002/
    └── ...
```

## Verify Dataset Setup

After downloading, you can verify the setup:

```bash
python3 -c "
from src.datasets.stanford_cars_dataset import get_stanford_cars
try:
    ds = get_stanford_cars('data/stanford_cars', 'train')
    print(f'✓ Dataset loaded successfully!')
    print(f'  Samples: {len(ds)}')
    print(f'  Classes: {ds.num_classes}')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

## Requirements

- `scipy` - Required for reading .mat annotation files
  ```bash
  pip install scipy
  ```

## Training

Once the dataset is set up, you can start training:

```bash
python train.py --config configs/stanford_cars.yaml
```

## Troubleshooting

### Issue: "Could not find train split directory"
- Ensure the dataset is downloaded and extracted
- Check that `car_ims/` directory exists in `data/stanford_cars/`
- Verify `cars_annos.mat` is present

### Issue: "Scipy is not found"
- Install scipy: `pip install scipy`

### Issue: "The original URL is broken"
- This is expected - torchvision's automatic download is broken
- You need to download the dataset manually as described above


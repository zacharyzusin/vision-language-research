#!/bin/bash
# Download script for iNaturalist 2021 dataset
# Based on official competition repository: https://github.com/visipedia/inat_comp/tree/master/2021

set -e  # Exit on error

# Configuration
DATA_ROOT="${1:-data/iNat2021}"
BASE_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/2021"

echo "=========================================="
echo "iNaturalist 2021 Dataset Download Script"
echo "=========================================="
echo ""
echo "Download directory: $DATA_ROOT"
echo ""

# Create directory
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

# Download annotation files
echo "Downloading annotation files..."
echo ""

# Training annotations
echo "1. Downloading train2021.json..."
if [ ! -f "train2021.json" ]; then
    wget -O train.json.tar.gz "${BASE_URL}/train.json.tar.gz"
    tar -xzf train.json.tar.gz
    rm train.json.tar.gz
    echo "   ✓ train2021.json downloaded"
else
    echo "   ✓ train2021.json already exists"
fi

# Validation annotations
echo "2. Downloading val2021.json..."
if [ ! -f "val2021.json" ]; then
    wget -O val.json.tar.gz "${BASE_URL}/val.json.tar.gz"
    tar -xzf val.json.tar.gz
    rm val.json.tar.gz
    echo "   ✓ val2021.json downloaded"
else
    echo "   ✓ val2021.json already exists"
fi

# Extract categories.json from train2021.json if needed
# (categories.json is typically embedded in the annotation files)
echo ""
echo "3. Checking for categories.json..."
if [ ! -f "categories.json" ]; then
    echo "   Note: categories.json should be extracted from train2021.json"
    echo "   You may need to extract it manually or it may be included in the archive"
fi

# Check available disk space
echo ""
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG "$DATA_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_SPACE=300  # GB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "⚠️  WARNING: Only ${AVAILABLE_SPACE}GB available, but ${REQUIRED_SPACE}GB+ needed"
    echo "   Consider downloading to a different location with more space"
    echo ""
fi

# Download training images (WARNING: This is ~300GB!)
echo ""
echo "=========================================="
echo "Image Downloads (LARGE FILES - ~300GB)"
echo "=========================================="
echo ""
echo "Training images: ~270GB (223GB compressed)"
echo "Validation images: ~10GB"
echo "Available space: ${AVAILABLE_SPACE}GB"
echo ""
read -p "Do you want to download the training images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if partial download exists
    if [ -f "train.tar.gz" ]; then
        echo "Found existing train.tar.gz (may be partial download)"
        read -p "Resume download? (y) or Delete and restart? (d) or Skip? (N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Dd]$ ]]; then
            rm -f train.tar.gz
            echo "Downloading training images (this may take a long time)..."
            wget -O train.tar.gz "${BASE_URL}/train.tar.gz"
        elif [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Resuming download of training images..."
            wget -c -O train.tar.gz "${BASE_URL}/train.tar.gz"
        else
            echo "Skipping training images download"
        fi
    else
        echo "Downloading training images (this may take a long time)..."
        wget -O train.tar.gz "${BASE_URL}/train.tar.gz"
    fi
    echo "Extracting training images..."
    tar -xzf train.tar.gz
    echo "   ✓ Training images extracted"
    # Optionally remove archive to save space
    read -p "Remove train.tar.gz to save space? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm train.tar.gz
    fi
fi

read -p "Do you want to download the validation images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading validation images..."
    wget -O val.tar.gz "${BASE_URL}/val.tar.gz"
    echo "Extracting validation images..."
    tar -xzf val.tar.gz
    echo "   ✓ Validation images extracted"
    # Optionally remove archive to save space
    read -p "Remove val.tar.gz to save space? (y/N): " -n 1 -r
    echo
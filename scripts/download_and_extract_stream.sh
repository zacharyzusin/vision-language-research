#!/bin/bash
# Download and extract iNaturalist 2021 images directly (streaming, no intermediate tar.gz)
# This saves disk space by not storing the compressed file

set -e

DATA_ROOT="${1:-data/iNat2021}"
BASE_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/2021"

cd "$DATA_ROOT"

echo "=========================================="
echo "Streaming Download & Extract (Saves Space)"
echo "=========================================="
echo ""
echo "This method downloads and extracts directly,"
echo "avoiding the need to store the 223GB tar.gz file"
echo ""

# Create extraction directory
mkdir -p 2021

echo "Downloading and extracting training images..."
echo "This will:"
echo "  1. Download from S3"
echo "  2. Extract directly to 2021/ directory"
echo "  3. Not save the tar.gz file (saves 223GB)"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Stream download and extract
echo "Starting download and extraction (this will take a long time)..."
wget -qO- "${BASE_URL}/train.tar.gz" | tar -xzf - -C 2021/

echo ""
echo "✓ Training images extracted to 2021/"
echo ""
echo "If you also want validation images:"
echo "  wget -qO- ${BASE_URL}/val.tar.gz | tar -xzf - -C 2021/"

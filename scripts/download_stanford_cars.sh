#!/bin/bash
# Download script for Stanford Cars dataset
# Tries multiple sources to download the dataset

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_ROOT="${1:-data/stanford_cars}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo -e "${GREEN}Stanford Cars Dataset Download Script${NC}"
echo "=========================================="
echo "Target directory: $(pwd)"
echo ""

# Function to download with wget
download_with_wget() {
    local url=$1
    local output=$2
    echo -e "${YELLOW}Attempting to download from: $url${NC}"
    if wget --progress=bar:force -O "$output" "$url" 2>&1 | grep -q "200 OK\|saved"; then
        return 0
    else
        return 1
    fi
}

# Function to download with curl
download_with_curl() {
    local url=$1
    local output=$2
    echo -e "${YELLOW}Attempting to download with curl from: $url${NC}"
    if curl -L --progress-bar -o "$output" "$url"; then
        return 0
    else
        return 1
    fi
}

# Function to verify file
verify_file() {
    local file=$1
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo -e "${GREEN}✓ Downloaded: $file ($(du -h "$file" | cut -f1))${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed: $file${NC}"
        return 1
    fi
}

# Try to download car_ims.tgz
echo -e "${GREEN}Downloading images (car_ims.tgz)...${NC}"
CAR_IMS_DOWNLOADED=0

# List of potential sources (in order of preference)
CAR_IMS_URLS=(
    "https://ai.stanford.edu/~jkrause/cars/car_ims.tgz"
    "https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/download?datasetVersionNumber=1"
    # Add more mirrors if available
)

for url in "${CAR_IMS_URLS[@]}"; do
    if download_with_wget "$url" "car_ims.tgz" || download_with_curl "$url" "car_ims.tgz"; then
        if verify_file "car_ims.tgz"; then
            CAR_IMS_DOWNLOADED=1
            break
        fi
    fi
    rm -f "car_ims.tgz"
done

if [ $CAR_IMS_DOWNLOADED -eq 0 ]; then
    echo -e "${RED}Failed to download car_ims.tgz from all sources${NC}"
    echo -e "${YELLOW}Please download manually from: https://ai.stanford.edu/~jkrause/cars/car_dataset.html${NC}"
    echo -e "${YELLOW}Or try: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset${NC}"
    exit 1
fi

# Try to download cars_annos.mat
echo ""
echo -e "${GREEN}Downloading annotations (cars_annos.mat)...${NC}"
ANNOS_DOWNLOADED=0

ANNO_URLS=(
    "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
    "https://ai.stanford.edu/~jkrause/cars/cars_annos.mat"
)

for url in "${ANNO_URLS[@]}"; do
    if [[ "$url" == *.tgz ]]; then
        # Download and extract devkit
        if download_with_wget "$url" "car_devkit.tgz" || download_with_curl "$url" "car_devkit.tgz"; then
            if [ -f "car_devkit.tgz" ] && [ -s "car_devkit.tgz" ]; then
                echo -e "${YELLOW}Extracting devkit...${NC}"
                tar -xzf car_devkit.tgz 2>/dev/null || true
                # Look for cars_annos.mat in extracted files
                if find . -name "cars_annos.mat" -type f | head -1 | xargs -I {} cp {} .; then
                    if verify_file "cars_annos.mat"; then
                        ANNOS_DOWNLOADED=1
                        rm -f car_devkit.tgz
                        break
                    fi
                fi
                rm -f car_devkit.tgz
            fi
        fi
    else
        # Direct download
        if download_with_wget "$url" "cars_annos.mat" || download_with_curl "$url" "cars_annos.mat"; then
            if verify_file "cars_annos.mat"; then
                ANNOS_DOWNLOADED=1
                break
            fi
        fi
        rm -f "cars_annos.mat"
    fi
done

if [ $ANNOS_DOWNLOADED -eq 0 ]; then
    echo -e "${YELLOW}Warning: Could not download cars_annos.mat automatically${NC}"
    echo -e "${YELLOW}Please download manually from: https://ai.stanford.edu/~jkrause/cars/car_dataset.html${NC}"
fi

# Try to download cars_meta.mat (optional)
echo ""
echo -e "${GREEN}Downloading metadata (cars_meta.mat) - optional...${NC}"
META_DOWNLOADED=0

META_URLS=(
    "https://ai.stanford.edu/~jkrause/cars/cars_meta.mat"
)

for url in "${META_URLS[@]}"; do
    if download_with_wget "$url" "cars_meta.mat" || download_with_curl "$url" "cars_meta.mat"; then
        if verify_file "cars_meta.mat"; then
            META_DOWNLOADED=1
            break
        fi
    fi
    rm -f "cars_meta.mat"
done

if [ $META_DOWNLOADED -eq 0 ]; then
    echo -e "${YELLOW}Note: cars_meta.mat not found (optional file)${NC}"
fi

# Extract images
echo ""
echo -e "${GREEN}Extracting images...${NC}"
if [ -f "car_ims.tgz" ]; then
    tar -xzf car_ims.tgz
    echo -e "${GREEN}✓ Images extracted${NC}"
    
    # Check if extraction created car_ims directory
    if [ ! -d "car_ims" ]; then
        # Maybe images were extracted to current directory
        if [ -d "cars_train" ] || [ -d "cars_test" ]; then
            echo -e "${YELLOW}Found alternative directory structure, creating car_ims/...${NC}"
            mkdir -p car_ims
            # Move images if needed
            find . -maxdepth 1 -name "*.jpg" -o -name "*.png" | head -5 | while read f; do
                if [ -f "$f" ]; then
                    mv "$f" car_ims/ 2>/dev/null || true
                fi
            done
        fi
    fi
fi

# Final verification
echo ""
echo -e "${GREEN}Verifying dataset structure...${NC}"
VERIFIED=1

if [ ! -d "car_ims" ]; then
    echo -e "${RED}✗ car_ims/ directory not found${NC}"
    VERIFIED=0
else
    IMG_COUNT=$(find car_ims -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    if [ "$IMG_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Found $IMG_COUNT images in car_ims/${NC}"
    else
        echo -e "${RED}✗ No images found in car_ims/${NC}"
        VERIFIED=0
    fi
fi

if [ ! -f "cars_annos.mat" ]; then
    echo -e "${RED}✗ cars_annos.mat not found (required)${NC}"
    VERIFIED=0
else
    echo -e "${GREEN}✓ cars_annos.mat found${NC}"
fi

if [ -f "cars_meta.mat" ]; then
    echo -e "${GREEN}✓ cars_meta.mat found${NC}"
fi

echo ""
if [ $VERIFIED -eq 1 ]; then
    echo -e "${GREEN}=========================================="
    echo -e "Dataset download and setup complete!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo "You can now run training with:"
    echo "  python train.py --config configs/stanford_cars.yaml"
    exit 0
else
    echo -e "${RED}=========================================="
    echo -e "Dataset setup incomplete!${NC}"
    echo -e "${RED}==========================================${NC}"
    echo ""
    echo "Please download missing files manually from:"
    echo "  https://ai.stanford.edu/~jkrause/cars/car_dataset.html"
    echo ""
    echo "Or try Kaggle:"
    echo "  https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset"
    exit 1
fi


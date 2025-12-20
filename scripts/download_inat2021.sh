#!/bin/bash
# Download script for iNaturalist 2021 competition dataset
# Based on official competition repository: https://github.com/visipedia/inat_comp/tree/master/2021
#
# Non-interactive behavior (cluster-friendly):
# - Always downloads annotations into DATA_ROOT
# - Extracts categories.json from train2021.json for stable label ordering
# - Downloads images only if DOWNLOAD_IMAGES=1
# - Optional streaming extract (no tar.gz stored) with STREAM_EXTRACT=1
#
# Expected resulting layout for this repo's loader (`src/datasets/inat_dataset.py`):
#   DATA_ROOT/train2021.json
#   DATA_ROOT/val2021.json
#   DATA_ROOT/categories.json
#   DATA_ROOT/2021/train/...
#   DATA_ROOT/2021/val/...

set -euo pipefail

DATA_ROOT="${1:-data/iNat2021}"
BASE_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/2021"

# Env flags (defaults are safe)
DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-0}"   # 1 to download images, 0 to skip
STREAM_EXTRACT="${STREAM_EXTRACT:-0}"    # 1 to stream-extract (no tar.gz stored)
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"      # 1 to keep train.tar.gz/val.tar.gz when not streaming

echo "=========================================="
echo "iNaturalist 2021 Dataset Download Script"
echo "=========================================="
echo ""
echo "Download directory: ${DATA_ROOT}"
echo "Download images:    ${DOWNLOAD_IMAGES}"
echo "Stream extract:     ${STREAM_EXTRACT}"
echo "Keep archives:      ${KEEP_ARCHIVES}"
echo ""

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo "Downloading annotation files..."
echo ""

if [ ! -f "train2021.json" ]; then
  echo "1) Downloading train2021.json ..."
    wget -O train.json.tar.gz "${BASE_URL}/train.json.tar.gz"
    tar -xzf train.json.tar.gz
  rm -f train.json.tar.gz
  # Official archive extracts to train.json; normalize to train2021.json for this repo.
  if [ -f "train.json" ] && [ ! -f "train2021.json" ]; then
    mv -f train.json train2021.json
  fi
else
  echo "1) train2021.json already exists"
fi

if [ ! -f "val2021.json" ]; then
  echo "2) Downloading val2021.json ..."
    wget -O val.json.tar.gz "${BASE_URL}/val.json.tar.gz"
    tar -xzf val.json.tar.gz
  rm -f val.json.tar.gz
  # Official archive extracts to val.json; normalize to val2021.json for this repo.
  if [ -f "val.json" ] && [ ! -f "val2021.json" ]; then
    mv -f val.json val2021.json
  fi
else
  echo "2) val2021.json already exists"
fi

echo ""
echo "3) Ensuring categories.json exists (extracted from train2021.json) ..."
if [ ! -f "categories.json" ]; then
  python - <<'PY'
import json
with open("train2021.json","r") as f:
    data=json.load(f)
cats=data.get("categories")
if not cats:
    raise SystemExit("train2021.json did not contain a 'categories' field; cannot extract categories.json")
with open("categories.json","w") as f:
    json.dump(cats,f)
print(f"Wrote categories.json with {len(cats)} categories")
PY
else
  echo "categories.json already exists"
fi

echo ""
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_SPACE=300  # GB (rough guideline)
if [ "${AVAILABLE_SPACE:-0}" -lt "${REQUIRED_SPACE}" ]; then
  echo "WARNING: Only ${AVAILABLE_SPACE}GB available, but ${REQUIRED_SPACE}GB+ recommended for full dataset."
fi

mkdir -p "2021"

if [ "${DOWNLOAD_IMAGES}" = "1" ]; then
  echo ""
  echo "Downloading and extracting images..."
  echo "  - train.tar.gz is ~223GB compressed"
  echo "  - val.tar.gz is ~9GB compressed"

  if [ "${STREAM_EXTRACT}" = "1" ]; then
echo ""
    echo "Streaming download + extract: train -> 2021/"
    wget -qO- "${BASE_URL}/train.tar.gz" | tar -xzf - -C 2021/
    echo "Done: train"

echo ""
    echo "Streaming download + extract: val -> 2021/"
    wget -qO- "${BASE_URL}/val.tar.gz" | tar -xzf - -C 2021/
    echo "Done: val"
  else
echo ""
    echo "Downloading train.tar.gz (resume enabled) ..."
            wget -c -O train.tar.gz "${BASE_URL}/train.tar.gz"
    echo "Extracting train -> 2021/ ..."
    tar -xzf train.tar.gz -C 2021/
    echo "Done: train"

    echo ""
    echo "Downloading val.tar.gz (resume enabled) ..."
    wget -c -O val.tar.gz "${BASE_URL}/val.tar.gz"
    echo "Extracting val -> 2021/ ..."
    tar -xzf val.tar.gz -C 2021/
    echo "Done: val"

    if [ "${KEEP_ARCHIVES}" != "1" ]; then
      rm -f train.tar.gz val.tar.gz
    fi
fi
else
  echo ""
  echo "Skipping image download (set DOWNLOAD_IMAGES=1 to download images)."
fi

echo ""
echo "Done."



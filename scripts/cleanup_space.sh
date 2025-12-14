#!/bin/bash
# Cleanup script to free up disk space

set -e

echo "=========================================="
echo "Disk Space Cleanup Script"
echo "=========================================="
echo ""

# Check current space
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Current available space: ${AVAILABLE}GB"
echo ""

# Option 1: Clean old checkpoints (keep only best and latest)
echo "1. Cleaning old checkpoints..."
echo "   Keeping: best.pt and best_epoch30.pt (latest)"
echo "   This will free ~25GB"
read -p "   Delete old checkpoints? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd checkpoints
    # Keep best.pt and the latest epoch
    KEEP_FILES="best.pt best_epoch30.pt"
    for file in *.pt; do
        if [[ ! " $KEEP_FILES " =~ " $file " ]]; then
            rm -f "$file"
            echo "     Deleted: $file"
        fi
    done
    cd ..
    echo "   ✓ Old checkpoints deleted"
else
    echo "   Skipped"
fi

# Option 2: Clean wandb logs
echo ""
echo "2. Cleaning wandb logs..."
echo "   This will free ~6GB"
read -p "   Delete wandb logs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf wandb/*
    echo "   ✓ Wandb logs deleted"
else
    echo "   Skipped"
fi

# Option 3: Clean text cache
echo ""
echo "3. Cleaning text cache..."
TEXT_CACHE_SIZE=$(du -sh text_cache 2>/dev/null | awk '{print $1}' || echo "0")
echo "   Text cache size: $TEXT_CACHE_SIZE"
read -p "   Delete text cache? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf text_cache/*
    echo "   ✓ Text cache deleted"
else
    echo "   Skipped"
fi

# Show final space
echo ""
echo "=========================================="
sleep 2
FINAL_AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Final available space: ${FINAL_AVAILABLE}GB"
echo "Space freed: $((FINAL_AVAILABLE - AVAILABLE))GB"
echo "=========================================="

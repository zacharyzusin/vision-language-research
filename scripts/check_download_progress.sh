#!/bin/bash
# Check progress of iNaturalist 2021 download

DATA_ROOT="${1:-data/iNat2021}"

echo "=========================================="
echo "Download Progress Check"
echo "=========================================="
echo ""

# Check if download is running
if pgrep -f "wget.*train.tar.gz" > /dev/null; then
    echo "✓ Download is running"
    echo ""
    # Show network activity
    echo "Network activity (last 10 seconds):"
    if command -v ifstat > /dev/null; then
        timeout 10 ifstat 1 1 2>/dev/null || echo "  (ifstat not available)"
    else
        echo "  (install ifstat for network monitoring: sudo apt install ifstat)"
    fi
else
    echo "Download process not found"
    echo "Check download_log.txt for status"
fi

echo ""
echo "Extracted files so far:"
if [ -d "$DATA_ROOT/2021" ]; then
    NUM_DIRS=$(find "$DATA_ROOT/2021" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    NUM_FILES=$(find "$DATA_ROOT/2021" -type f 2>/dev/null | wc -l)
    SIZE=$(du -sh "$DATA_ROOT/2021" 2>/dev/null | awk '{print $1}')
    echo "  Directories: $NUM_DIRS"
    echo "  Files: $NUM_FILES"
    echo "  Size: $SIZE"
else
    echo "  No files extracted yet"
fi

echo ""
echo "Disk space:"
df -h "$DATA_ROOT" | tail -1

echo ""
echo "To view download log:"
echo "  tail -f download_log.txt"

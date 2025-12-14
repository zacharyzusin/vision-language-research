#!/bin/bash
# Monitor validation images download progress

DATA_ROOT="data/iNat2021/2021/val"
TARGET_SIZE=10  # GB (approximate final size)
TARGET_FILES=100000  # Approximate number of files

while true; do
    clear
    echo "=========================================="
    echo "Validation Images Download Progress"
    echo "=========================================="
    echo ""
    
    # Check if download is running
    if pgrep -f "wget.*val.tar.gz" > /dev/null; then
        echo "✓ Download Status: RUNNING"
    else
        echo "✗ Download Status: NOT RUNNING"
        echo "  (May have completed or failed)"
    fi
    
    echo ""
    echo "--- Progress ---"
    
    if [ -d "$DATA_ROOT" ]; then
        CURRENT_SIZE=$(du -sm "$DATA_ROOT" 2>/dev/null | awk '{print $1}')
        CURRENT_SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $CURRENT_SIZE / 1024}")
        PERCENT=$(awk "BEGIN {percent = ($CURRENT_SIZE_GB / $TARGET_SIZE) * 100; if (percent < 0) percent = 0; if (percent > 100) percent = 100; printf \"%.1f\", percent}")
        
        NUM_FILES=$(find "$DATA_ROOT" -type f 2>/dev/null | wc -l)
        FILES_PERCENT=$(awk "BEGIN {percent = ($NUM_FILES / $TARGET_FILES) * 100; if (percent < 0) percent = 0; if (percent > 100) percent = 100; printf \"%.1f\", percent}")
        
        echo "Size: ${CURRENT_SIZE_GB} GB / ~${TARGET_SIZE} GB (${PERCENT}%)"
        echo "Files: $(printf "%'d" $NUM_FILES) / ~$(printf "%'d" $TARGET_FILES) (${FILES_PERCENT}%)"
        
        # Progress bar
        BAR_LENGTH=50
        FILLED=$(awk "BEGIN {filled = int(($PERCENT / 100) * $BAR_LENGTH); if (filled < 0) filled = 0; if (filled > $BAR_LENGTH) filled = $BAR_LENGTH; print filled}")
        
        BAR=""
        i=0
        while [ $i -lt $BAR_LENGTH ]; do
            if [ $i -lt $FILLED ]; then
                BAR="${BAR}█"
            else
                BAR="${BAR}░"
            fi
            i=$((i + 1))
        done
        echo "[$BAR] ${PERCENT}%"
    else
        echo "Waiting for extraction to start..."
    fi
    
    echo ""
    echo "--- System ---"
    df -h data/iNat2021 2>/dev/null | tail -1 | awk '{print "Disk: " $4 " free (" $5 " used)"}'
    
    # Process info
    if pgrep -f "wget.*val" > /dev/null; then
        WGET_CPU=$(ps aux | grep "wget.*val" | grep -v grep | head -1 | awk '{print $3}')
        TAR_CPU=$(ps aux | grep "tar.*val" | grep -v grep | head -1 | awk '{print $3}')
        if [ ! -z "$WGET_CPU" ]; then
            echo "CPU Usage: wget=${WGET_CPU}%"
            if [ ! -z "$TAR_CPU" ]; then
                echo "            tar=${TAR_CPU}%"
            fi
        fi
    fi
    
    echo ""
    echo "--- Time ---"
    echo "Current: $(date '+%H:%M:%S')"
    
    echo ""
    echo "Press Ctrl+C to exit"
    echo "Refreshing every 3 seconds..."
    
    sleep 3
done

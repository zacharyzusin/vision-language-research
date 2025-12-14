#!/bin/bash
# Real-time download monitoring script

DATA_ROOT="data/iNat2021/2021"
TARGET_SIZE=270  # GB (approximate final size)

while true; do
    clear
    echo "=========================================="
    echo "iNaturalist 2021 Download Progress"
    echo "=========================================="
    echo ""
    
    # Check if download is running
    if pgrep -f "wget.*train.tar.gz" > /dev/null; then
        echo "✓ Download Status: RUNNING"
    else
        echo "✗ Download Status: NOT RUNNING"
        echo "  (May have completed or failed)"
    fi
    
    echo ""
    echo "--- Progress ---"
    
    # Current size
    if [ -d "$DATA_ROOT" ]; then
        CURRENT_SIZE=$(du -sm "$DATA_ROOT" 2>/dev/null | awk '{print $1}')
        
        # Calculate using awk for reliability
        CURRENT_SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $CURRENT_SIZE / 1024}")
        PERCENT=$(awk "BEGIN {percent = ($CURRENT_SIZE_GB / $TARGET_SIZE) * 100; if (percent < 0) percent = 0; if (percent > 100) percent = 100; printf \"%.1f\", percent}")
        
        echo "Extracted: ${CURRENT_SIZE_GB} GB / ~${TARGET_SIZE} GB"
        echo "Progress: ${PERCENT}%"
        
        # Progress bar - calculate filled blocks using awk
        BAR_LENGTH=50
        FILLED=$(awk "BEGIN {filled = int(($PERCENT / 100) * $BAR_LENGTH); if (filled < 0) filled = 0; if (filled > $BAR_LENGTH) filled = $BAR_LENGTH; print filled}")
        
        # Build progress bar
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
    echo "--- Statistics ---"
    
    if [ -d "$DATA_ROOT" ]; then
        NUM_FILES=$(find "$DATA_ROOT" -type f 2>/dev/null | wc -l)
        NUM_DIRS=$(find "$DATA_ROOT" -type d 2>/dev/null | wc -l)
        echo "Files: $(printf "%'d" $NUM_FILES)"
        echo "Directories: $(printf "%'d" $NUM_DIRS)"
    fi
    
    echo ""
    echo "--- System ---"
    df -h . | tail -1 | awk '{print "Disk: " $4 " free (" $5 " used)"}'
    
    # Process info
    if pgrep -f "wget.*train" > /dev/null; then
        WGET_CPU=$(ps aux | grep "wget.*train" | grep -v grep | awk '{print $3}' | head -1)
        TAR_CPU=$(ps aux | grep "tar.*2021" | grep -v grep | head -1 | awk '{print $3}')
        if [ ! -z "$WGET_CPU" ] && [ ! -z "$TAR_CPU" ]; then
            echo "CPU Usage: wget=${WGET_CPU}% tar=${TAR_CPU}%"
        fi
    fi
    
    echo ""
    echo "--- Time ---"
    echo "Current: $(date '+%H:%M:%S')"
    if [ -f download_log.txt ]; then
        START_TIME=$(grep "Starting download" download_log.txt | tail -1 | awk '{print $6, $7, $8}')
        if [ ! -z "$START_TIME" ]; then
            echo "Started: $START_TIME"
        fi
    fi
    
    echo ""
    echo "Press Ctrl+C to exit"
    echo "Refreshing every 5 seconds..."
    
    sleep 5
done

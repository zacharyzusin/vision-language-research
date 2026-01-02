#!/bin/bash
# Quick script to check job outputs

echo "=== Job Status ==="
squeue -u zwz2000 | grep -E "JOBID|sweep_cars" | head -10

echo ""
echo "=== Latest Log Files ==="
ls -lt logs/sweep_cars_*.out 2>/dev/null | head -5 | awk '{print $NF}'

echo ""
echo "=== Recent Activity (last 10 lines of most recent log) ==="
LATEST=$(ls -t logs/sweep_cars_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "File: $LATEST"
    tail -10 "$LATEST"
fi

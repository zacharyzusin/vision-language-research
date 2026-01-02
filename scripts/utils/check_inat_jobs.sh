#!/bin/bash
# Check iNaturalist sweep job outputs

echo "=== iNaturalist Sweep Jobs Status ==="
squeue -u zwz2000 -j 6046474,6046475,6046476,6046477,6046478,6046479,6046480 -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %.20R" 2>/dev/null

echo ""
echo "=== Training Progress ==="
for job in 6046474 6046475 6046476 6046477 6046478 6046479 6046480; do
    echo ""
    echo "Job $job:"
    if [ -f "logs/sweep_1node_${job}.out" ]; then
        tail -3 "logs/sweep_1node_${job}.out" | tail -2
    else
        status=$(squeue -j $job -h -o "%T %R" 2>/dev/null | awk '{print $1}')
        echo "  Status: $status (log not created yet)"
    fi
done

echo ""
echo "=== Recent Epoch/Validation Results (if any) ==="
for job in 6046474 6046475 6046476 6046477 6046478 6046479 6046480; do
    if [ -f "logs/sweep_1node_${job}.out" ]; then
        results=$(grep -E "Epoch [0-9]+:|Val Acc|best_val" "logs/sweep_1node_${job}.out" | tail -3)
        if [ -n "$results" ]; then
            echo "Job $job:"
            echo "$results"
        fi
    fi
done

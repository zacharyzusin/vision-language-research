#!/bin/bash
# Script to initialize wandb sweep for iNaturalist (TEST - quick verification)
# Usage: ./launch_sweep_test.sh

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Number of agents (default: 2 for parallel testing)
NUM_AGENTS=${1:-2}

echo "=== Initializing W&B Test Sweep for iNaturalist ==="
echo "Sweep config: sweep_k_values_test.yaml"
echo "K values: [1, 2] (quick test)"
echo "Agents: $NUM_AGENTS (parallel testing)"
echo "GPUs per agent: 1 (single GPU, no DDP)"
echo ""

# Initialize the sweep and get the sweep ID
SWEEP_ID=$(wandb sweep sweep_k_values_test.yaml 2>&1 | grep -oP 'wandb agent [^\s]+' | awk '{print $3}' || echo "")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to initialize sweep. Make sure wandb is configured."
    echo "Run: wandb login"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Sweep URL: https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/inat-mop-clip/sweeps/$SWEEP_ID"
echo ""

# Submit agents for parallel testing
echo "=== Submitting Test Slurm Jobs ==="
for i in $(seq 1 $NUM_AGENTS); do
    echo "Submitting agent $i/$NUM_AGENTS..."
    
    JOB_ID=$(sbatch \
        --export=WANDB_SWEEP_ID=$SWEEP_ID \
        --job-name="sweep_test_${i}" \
        slurm/active/train_mop_ddp_sweep_test_1gpu.sh 2>&1 | grep -oP '\d+' | head -1)
    
    if [ -n "$JOB_ID" ]; then
        echo "  Agent $i submitted as job ID: $JOB_ID"
    else
        echo "  ERROR: Failed to submit agent $i"
    fi
done

echo ""
echo "=== Summary ==="
echo "Sweep ID: $SWEEP_ID"
echo "Number of agents: $NUM_AGENTS"
echo "Dataset: iNaturalist"
echo "Configuration: Single-GPU per agent (1 GPU), 2 K values [1, 2]"
echo "Expected: 2 agents will run K=1 and K=2 in parallel"
echo ""
echo "Monitor sweep progress at:"
echo "  https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/inat-mop-clip/sweeps/$SWEEP_ID"
echo ""
echo "Check job status:"
echo "  squeue -u $(whoami) | grep sweep_test"
echo ""
echo "View logs:"
echo "  tail -f logs/sweep_test_1gpu_*.out"


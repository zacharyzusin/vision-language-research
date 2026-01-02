#!/bin/bash
# Script to initialize wandb sweep for KikiBouba v1 (1 GPU) and submit Slurm job
# Usage: ./launch_sweep_kikibouba_v1_1gpu.sh

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

echo "=== Initializing W&B Sweep for KikiBouba v1 (1 GPU) ==="
echo "Sweep config: sweep_k_values_kikibouba_v1_1gpu.yaml"
echo ""

# Initialize the sweep and get the sweep ID
SWEEP_ID=$(wandb sweep sweep_k_values_kikibouba_v1_1gpu.yaml 2>&1 | grep -oP 'wandb agent [^\s]+' | awk '{print $3}' || echo "")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to initialize sweep. Make sure wandb is configured."
    echo "Run: wandb login"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Sweep URL: https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/kikibouba-v1-mop-clip/sweeps/$SWEEP_ID"
echo ""

# Submit Slurm job
echo "=== Submitting Slurm Job for Sweep Agent ==="
JOB_ID=$(sbatch \
    --export=WANDB_SWEEP_ID=$SWEEP_ID \
    --job-name="sweep_kb1_1gpu" \
    slurm/active/train_mop_ddp_sweep_kikibouba_v1_1gpu.sh 2>&1 | grep -oP '\d+' | head -1)

if [ -n "$JOB_ID" ]; then
    echo "  Agent submitted as job ID: $JOB_ID"
else
    echo "  ERROR: Failed to submit agent"
    exit 1
fi

echo ""
echo "=== Summary ==="
echo "Sweep ID: $SWEEP_ID"
echo "Job ID: $JOB_ID"
echo "Dataset: KikiBouba v1"
echo "Hardware: 1 GPU"
echo ""
echo "Monitor sweep progress at:"
echo "  https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/kikibouba-v1-mop-clip/sweeps/$SWEEP_ID"
echo ""
echo "Check job status:"
echo "  squeue -u $(whoami)"
echo ""
echo "View logs:"
echo "  tail -f logs/sweep_kb1_1gpu_${JOB_ID}.out"


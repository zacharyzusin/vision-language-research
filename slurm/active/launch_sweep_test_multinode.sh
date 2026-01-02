#!/bin/bash
# Script to initialize wandb sweep for iNaturalist (TEST - multi-node verification)
# Usage: ./launch_sweep_test_multinode.sh

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

echo "=== Initializing W&B Test Sweep for iNaturalist (Multi-Node) ==="
echo "Sweep config: sweep_k_values_test_multinode.yaml"
echo "K values: [1, 2] (quick test)"
echo "Agents: 1 (test one agent first)"
echo "GPUs per agent: 16 (2 nodes × 8 GPUs)"
echo "Time limit: 2 hours (just to verify setup works)"
echo ""

# Initialize the sweep and get the sweep ID
SWEEP_ID=$(wandb sweep sweep_k_values_test_multinode.yaml 2>&1 | grep -oP 'wandb agent [^\s]+' | awk '{print $3}' || echo "")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to initialize sweep. Make sure wandb is configured."
    echo "Run: wandb login"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Sweep URL: https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/inat-mop-clip/sweeps/$SWEEP_ID"
echo ""

# Submit 1 agent for quick test
echo "=== Submitting Test Slurm Job ==="
JOB_ID=$(sbatch \
    --export=WANDB_SWEEP_ID=$SWEEP_ID \
    --job-name="sweep_test_mn" \
    slurm/active/train_mop_ddp_sweep_test_multinode.sh 2>&1 | grep -oP '\d+' | head -1)

if [ -n "$JOB_ID" ]; then
    echo "  Test agent submitted as job ID: $JOB_ID"
else
    echo "  ERROR: Failed to submit test agent"
    exit 1
fi

echo ""
echo "=== Summary ==="
echo "Sweep ID: $SWEEP_ID"
echo "Job ID: $JOB_ID"
echo "Dataset: iNaturalist"
echo "Configuration: Multi-node (2 nodes, 16 GPUs), 2 K values [1, 2]"
echo "Purpose: Verify multi-node DDP setup works before full sweep"
echo ""
echo "Monitor sweep progress at:"
echo "  https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/inat-mop-clip/sweeps/$SWEEP_ID"
echo ""
echo "Check job status:"
echo "  squeue -u $(whoami) | grep $JOB_ID"
echo ""
echo "View logs:"
echo "  tail -f logs/sweep_test_mn_${JOB_ID}.out"
echo ""
echo "What to verify:"
echo "  1. Job starts and gets 2 nodes"
echo "  2. Both nodes launch torchrun successfully"
echo "  3. Training starts (you'll see epoch logs)"
echo "  4. DDP communication works (no errors about master address/port)"
echo "  5. W&B logs appear correctly"


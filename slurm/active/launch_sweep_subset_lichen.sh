#!/bin/bash
# Script to initialize wandb sweep for iNaturalist Lichen subset and submit Slurm jobs to run sweep agents
# Usage: ./launch_sweep_subset_lichen.sh [num_agents]

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Number of sweep agents to run (default: 1, can run multiple in parallel)
NUM_AGENTS=${1:-1}

echo "=== Initializing W&B Sweep for iNaturalist Lichen Subset ==="
echo "Sweep config: sweep_k_values_subset_lichen.yaml"
echo "Number of agents: $NUM_AGENTS"
echo ""

# Initialize the sweep and get the sweep ID
SWEEP_ID=$(wandb sweep sweep_k_values_subset_lichen.yaml 2>&1 | grep -oP 'wandb agent [^\s]+' | awk '{print $3}' || echo "")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to initialize sweep. Make sure wandb is configured."
    echo "Run: wandb login"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Sweep URL: https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/inat-subset-lichen-mop-clip/sweeps/$SWEEP_ID"
echo ""

# Submit Slurm jobs for each agent
echo "=== Submitting Slurm Jobs for Sweep Agents ==="
for i in $(seq 1 $NUM_AGENTS); do
    echo "Submitting agent $i/$NUM_AGENTS..."
    
    # Submit job with sweep ID as environment variable (using 1 GPU for faster scheduling)
    JOB_ID=$(sbatch \
        --export=WANDB_SWEEP_ID=$SWEEP_ID \
        --job-name="sweep_lichen_agent_${i}" \
        slurm/active/train_mop_ddp_sweep_subset_lichen.sh 2>&1 | grep -oP '\d+' | head -1)
    
    if [ -n "$JOB_ID" ]; then
        echo "  Agent $i submitted as job ID: $JOB_ID"
    else
        echo "  ERROR: Failed to submit agent $i"
    fi
done

echo ""
echo "=== Summary ==="
echo "Sweep ID: $SWEEP_ID"
echo "Number of agents submitted: $NUM_AGENTS"
echo "Dataset: iNaturalist Lichen Subset (6 species)"
echo "Configuration: Single-GPU (1 GPU)"
echo ""
echo "Monitor sweep progress at:"
echo "  https://wandb.ai/$(wandb whoami 2>/dev/null || echo 'your-entity')/inat-subset-lichen-mop-clip/sweeps/$SWEEP_ID"
echo ""
echo "Check job status:"
echo "  squeue -u $(whoami)"
echo ""
echo "View logs:"
echo "  tail -f logs/sweep_lichen_1gpu_*.out"


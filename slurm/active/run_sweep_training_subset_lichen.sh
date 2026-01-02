#!/bin/bash
# Wrapper script to launch single-GPU training for wandb sweep (iNaturalist Lichen subset)
# This script is called by wandb agent and launches training

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Activate conda environment (needed when called from wandb agent)
module load anaconda
module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate mop_clip

# Single-GPU setup (no DDP)
echo "Single-GPU configuration (iNaturalist Lichen subset):"
echo "  Number of GPUs: 1"
echo "  No DDP (single process)"

echo "Launching training for Lichen subset sweep run..."
echo "  GPU: 1"

# Single GPU: just run python directly (no torchrun)
python train.py --config configs/subset_lichen.yaml


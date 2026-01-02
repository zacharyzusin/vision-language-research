#!/bin/bash
# Wrapper script to launch single-node DDP training for wandb sweep (KikiBouba v2)
# This script is called by wandb agent and launches torchrun

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Activate conda environment (needed when called from wandb agent)
module load anaconda
module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate mop_clip

# Single-node DDP setup using torchrun
GPUS_PER_NODE=8

echo "Single-node DDP configuration (KikiBouba v2):"
echo "  Number of nodes: 1"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $GPUS_PER_NODE"

# Use standalone mode for single-node (no need for master address/port)
# Slurm -> torchrun settings
NPROC="${SLURM_GPUS_ON_NODE:-8}"

echo "Launching torchrun for KikiBouba v2 sweep run..."
echo "  GPUs: $NPROC"

# Single-node: use standalone mode (simpler, no need for master address/port)
torchrun --standalone --nproc_per_node="${NPROC}" train.py --config configs/kikibouba_v2.yaml


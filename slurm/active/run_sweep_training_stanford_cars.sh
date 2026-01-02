#!/bin/bash
# Wrapper script to launch single-node DDP training for wandb sweep (Stanford Cars)
# This script is called by wandb agent and launches torchrun

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Single-node DDP setup using torchrun
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# Get master node address from Slurm (or use localhost for single node)
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
else
    MASTER_ADDR="localhost"
fi
MASTER_PORT=29500

# Set DDP environment variables for PyTorch
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo "Launching torchrun for Stanford Cars sweep run..."
echo "  Node: $(hostname)"
echo "  GPUs: $TOTAL_GPUS (4 per node)"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"

# Single-node: launch torchrun directly (no srun needed)
torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --config configs/stanford_cars.yaml


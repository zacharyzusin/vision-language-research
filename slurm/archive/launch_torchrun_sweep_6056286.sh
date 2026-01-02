#!/bin/bash
set -e

# Activate conda environment in launch script (needed for srun)
module load anaconda
module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate mop_clip

# Change to correct directory
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Calculate this node's rank
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE_RANK=0
for i in "${!NODE_LIST[@]}"; do
    if [ "${NODE_LIST[$i]}" = "$(hostname)" ]; then
        NODE_RANK=$i
        break
    fi
done

echo "Node $(hostname) (rank $NODE_RANK) launching torchrun for sweep run..."
echo "  Working directory: $(pwd)"
echo "  Master: ins088:29500"
echo "  Node rank: $NODE_RANK"
echo "  Total nodes: $SLURM_NNODES"

# Ensure torchrun is available
which torchrun || { echo "ERROR: torchrun not found in PATH"; exit 127; }

# Launch torchrun
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=ins088 \
    --master_port=29500 \
    train.py --config configs/default.yaml

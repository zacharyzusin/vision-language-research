#!/bin/bash
# Wrapper script to launch multi-node DDP training for wandb sweep (KikiBouba v2)
# This script is called by wandb agent and launches torchrun

set -euo pipefail

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Multi-node DDP setup using torchrun
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# Get master node address from Slurm
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Set DDP environment variables for PyTorch
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Create a helper script that calculates node rank and launches torchrun
LAUNCH_SCRIPT="${PWD}/launch_torchrun_sweep_kikibouba_v2_${SLURM_JOB_ID}.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCH_EOF'
#!/bin/bash
# Calculate this node's rank
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE_RANK=0
for i in "${!NODE_LIST[@]}"; do
    if [ "${NODE_LIST[$i]}" = "$(hostname)" ]; then
        NODE_RANK=$i
        break
    fi
done

echo "Node $(hostname) (rank $NODE_RANK) launching torchrun for KikiBouba v2 sweep run..."

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --config configs/kikibouba_v2.yaml
LAUNCH_EOF

chmod +x "$LAUNCH_SCRIPT"

# Launch one task per node, each running torchrun with 8 local processes
srun --ntasks=$NNODES --ntasks-per-node=1 "$LAUNCH_SCRIPT"

# Cleanup
rm -f "$LAUNCH_SCRIPT"


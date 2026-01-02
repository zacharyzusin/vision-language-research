#!/bin/bash
#SBATCH --account=edu                 # Your account (use 'edu' for your course)
#SBATCH --job-name=mop_clip_ddp       # Job name
#SBATCH --partition=short             # A6000 GPUs are available on short (edu1 has no GPUs)
#SBATCH --nodes=2                     # Use 2 nodes (8 GPUs each = 16 total)
#SBATCH --ntasks=2                    # One task per node (torchrun spawns 8 processes per node)
#SBATCH --ntasks-per-node=1           # One task per node
#SBATCH --cpus-per-task=16            # CPU cores per task (torchrun needs CPUs for 8 processes)
#SBATCH --mem=128gb                   # Memory per node
#SBATCH --time=12:00:00               # Time limit (edu1 MaxTime is 12 hours)
#SBATCH --gres=gpu:A6000:8            # Request 8x A6000 GPUs per node (16 total)
#SBATCH --exclude=ins082              # ins082 has a bad GPU device handle (torch CUDA init fails)
#SBATCH --output=logs/train_ddp_%j.out # Output file (%j = job ID)
#SBATCH --error=logs/train_ddp_%j.err  # Error file

set -euo pipefail

mkdir -p logs

module load anaconda
module load cuda/12.3

# Robust conda activation inside non-interactive Slurm shells
eval "$(conda shell.bash hook)"
conda activate mop_clip

echo "Python environment:"
echo "  which python: $(which python)"
python -V
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
try:
    import wandb
    print("wandb:", wandb.__version__)
except Exception as e:
    print("wandb import failed:", e)
PY

echo "Job started at: $(date)"
echo "Running on nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPU info:"
nvidia-smi

# W&B online logging (streams metrics to the dashboard)
export WANDB_MODE=online

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Fail fast if images aren't present
if [ ! -d "data/iNat2021/2021/train" ] || [ ! -d "data/iNat2021/2021/val" ]; then
  echo "ERROR: iNat2021 images not found under data/iNat2021/2021/{train,val}"
  exit 2
fi

# Multi-node DDP setup using torchrun (same approach as single-node, but with --nnodes)
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

echo "Multi-node DDP configuration:"
echo "  Number of nodes: $NNODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Node list: $SLURM_JOB_NODELIST"

# Sanity check: confirm PyTorch sees the allocated GPUs
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "nvidia-smi -L:"
nvidia-smi -L || true
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

# Get master node address from Slurm
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Set DDP environment variables for PyTorch (torchrun will use these)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo "Multi-node DDP launch configuration:"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Number of nodes: $NNODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $TOTAL_GPUS"

# Multi-node DDP with torchrun
# Strategy: Use srun to launch torchrun on each node (one task per node)
# Each node runs torchrun locally with 8 processes, and they coordinate via MASTER_ADDR
# This is simpler and more reliable than trying to coordinate torchrun across nodes

# Create a helper script that calculates node rank and launches torchrun
# Use project directory (shared across nodes) instead of /tmp (node-local)
LAUNCH_SCRIPT="${PWD}/launch_torchrun_multinode_${SLURM_JOB_ID}.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCH_EOF'
#!/bin/bash
# Activate conda environment in launch script (needed for srun)
module load anaconda
module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate mop_clip

# Calculate this node's rank
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE_RANK=0
for i in "${!NODE_LIST[@]}"; do
    if [ "${NODE_LIST[$i]}" = "$(hostname)" ]; then
        NODE_RANK=$i
        break
    fi
done

echo "Node $(hostname) (rank $NODE_RANK) launching torchrun..."

# torchrun should be in PATH after conda activation
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --config configs/default.yaml
LAUNCH_EOF

chmod +x "$LAUNCH_SCRIPT"

echo "Launching torchrun on all nodes..."
echo "Launch script: $LAUNCH_SCRIPT"
# Launch one task per node, each running torchrun with 8 local processes
# Pass environment variables explicitly so they're available in the launch script
srun --ntasks=$NNODES --ntasks-per-node=1 \
    --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT \
    "$LAUNCH_SCRIPT"

# Cleanup
rm -f "$LAUNCH_SCRIPT"

echo "Job finished at: $(date)"


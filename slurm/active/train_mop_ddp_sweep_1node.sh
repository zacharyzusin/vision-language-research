#!/bin/bash
#SBATCH --account=edu                 # Your account
#SBATCH --job-name=mop_clip_sweep    # Job name
#SBATCH --partition=short             # A6000 GPUs are available on short
#SBATCH --nodes=1                     # Use 1 node (8 GPUs total)
#SBATCH --ntasks=1                    # One task (torchrun spawns 8 processes)
#SBATCH --cpus-per-task=16            # CPU cores per task
#SBATCH --mem=128gb                   # Memory per node
#SBATCH --time=12:00:00               # Time limit (12 hours)
#SBATCH --gres=gpu:A6000:8            # Request 8x A6000 GPUs (max per node)
#SBATCH --exclude=ins082              # ins082 has a bad GPU device handle
#SBATCH --output=logs/sweep_1node_%j.out     # Output file (%j = job ID)
#SBATCH --error=logs/sweep_1node_%j.err     # Error file

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
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# W&B online logging
export WANDB_MODE=online

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Fail fast if images aren't present
if [ ! -d "data/iNat2021/2021/train" ] || [ ! -d "data/iNat2021/2021/val" ]; then
  echo "ERROR: iNat2021 images not found under data/iNat2021/2021/{train,val}"
  exit 2
fi

# Get sweep ID from environment variable (set by launch script)
SWEEP_ID="${WANDB_SWEEP_ID}"
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: WANDB_SWEEP_ID environment variable not set"
    echo "This script should be launched via the sweep launcher script"
    exit 1
fi

echo "Running wandb sweep agent for sweep ID: $SWEEP_ID"

# Single-node DDP setup using torchrun
GPUS_PER_NODE=8

echo "Single-node DDP configuration:"
echo "  Number of nodes: 1"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $GPUS_PER_NODE"

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

# Run wandb agent on main process only
# The agent will call the script specified in sweep YAML for each sweep run, which launches torchrun
echo "Launching wandb agent (sweep ID: $SWEEP_ID)..."
echo "Agent will run training jobs using script specified in sweep YAML"

# Run wandb agent - it will execute the script specified in the sweep YAML for each sweep run
# The wrapper script handles single-node DDP setup
# By default, agent runs until sweep is complete (all K values)
echo "Starting wandb agent (will run all sweep runs sequentially)..."
wandb agent $SWEEP_ID

echo "Job finished at: $(date)"


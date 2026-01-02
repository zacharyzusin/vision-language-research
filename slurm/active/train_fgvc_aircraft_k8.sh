#!/bin/bash
#SBATCH --account=edu                 # Your account
#SBATCH --job-name=mop_aircraft_k8   # Job name
#SBATCH --partition=short             # A6000 GPUs are available on short
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --ntasks=1                    # One task (torchrun spawns processes)
#SBATCH --ntasks-per-node=1           # One task per node
#SBATCH --cpus-per-task=8             # Reduced from 16 - can work with less
#SBATCH --mem=64gb                    # Reduced from 128gb - more flexible
#SBATCH --time=12:00:00               # Time limit (12 hours)
#SBATCH --gres=gpu:2                  # Request 2 GPUs (more flexible than 4x A6000)
#SBATCH --exclude=ins082              # ins082 has a bad GPU device handle
#SBATCH --output=logs/train_fgvc_aircraft_k8_%j.out     # Output file (%j = job ID)
#SBATCH --error=logs/train_fgvc_aircraft_k8_%j.err     # Error file

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
    if torch.cuda.is_available():
        print("CUDA devices:", torch.cuda.device_count())
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

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Set environment variables for DDP
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Detect number of GPUs available
NPROC=${SLURM_GPUS_ON_NODE:-2}
echo "Detected GPUs: $NPROC"

echo ""
echo "=== Starting training with K=8 ==="
echo "Config: configs/fgvc_aircraft.yaml"
echo "GPUs: $NPROC"
echo ""

# Launch training with torchrun (handles DDP setup)
torchrun --standalone --nproc_per_node="${NPROC}" \
    train.py \
    --config configs/fgvc_aircraft.yaml

echo ""
echo "Job finished at: $(date)"


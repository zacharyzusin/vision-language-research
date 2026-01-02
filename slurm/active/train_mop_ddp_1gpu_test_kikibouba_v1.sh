#!/bin/bash
#SBATCH --account=edu                 # Your account
#SBATCH --job-name=mop_kb1_test       # Job name
#SBATCH --partition=short              # A6000 GPUs are available on short
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=4              # CPU cores per task (reduced for test)
#SBATCH --mem=32gb                     # Memory (reduced for test)
#SBATCH --time=12:00:00                # Time limit (12 hours to complete full training)
#SBATCH --gres=gpu:A6000:1             # Request 1x A6000 GPU for testing
#SBATCH --exclude=ins082               # ins082 has a bad GPU device handle
#SBATCH --output=logs/train_test_kb1_1gpu_%j.out # Output file
#SBATCH --error=logs/train_test_kb1_1gpu_%j.err  # Error file

set -euo pipefail

mkdir -p logs

module load anaconda
module load cuda/12.3

# Robust conda activation inside non-interactive Slurm shells
eval "$(conda shell.bash hook)"
conda activate mop_clip

echo "=== TEST JOB: 1 GPU (KikiBouba v1) ==="
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

# Fail fast if dataset isn't present
if [ ! -d "data/kikibouba/kiki_bouba_v1_split/train" ] || [ ! -d "data/kikibouba/kiki_bouba_v1_split/val" ]; then
  echo "ERROR: KikiBouba v1 dataset not found under data/kikibouba/kiki_bouba_v1_split/{train,val}"
  exit 2
fi

# Sanity check: confirm PyTorch sees the allocated GPU
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

# Single GPU: use torchrun with nproc_per_node=1 (no DDP, but uses same code path)
echo "=== Starting training on 1 GPU (KikiBouba v1) ==="
torchrun --standalone --nproc_per_node=1 train.py --config configs/kikibouba_v1.yaml

echo "Job finished at: $(date)"


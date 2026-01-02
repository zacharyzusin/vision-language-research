#!/bin/bash
#SBATCH --account=edu                 # Your account (use 'edu' for your course)
#SBATCH --job-name=mop_clip_ddp       # Job name
#SBATCH --partition=short             # A6000 GPUs are available on short (edu1 has no GPUs)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # One task; torchrun spawns processes
#SBATCH --cpus-per-task=16            # CPU cores per task
#SBATCH --mem=128gb                   # Memory per node (increased from 64gb due to OOM)
#SBATCH --time=12:00:00               # Time limit (edu1 MaxTime is 12 hours)
#SBATCH --gres=gpu:A6000:16           # Request 16x A6000 GPUs (will get ~27 epochs in 12h)
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
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# W&B online logging (streams metrics to the dashboard)
# If you don't have outbound internet on compute nodes, set this back to "offline".
export WANDB_MODE=online

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Fail fast if images aren't present (annotations-only setup will crash later)
if [ ! -d "data/iNat2021/2021/train" ] || [ ! -d "data/iNat2021/2021/val" ]; then
  echo "ERROR: iNat2021 images not found under data/iNat2021/2021/{train,val}"
  echo "Run first:"
  echo "  DOWNLOAD_IMAGES=1 STREAM_EXTRACT=1 ./scripts/download_inat2021.sh data/iNat2021"
  exit 2
fi

# Sanity check: confirm PyTorch sees the allocated GPUs in the job log
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

# Slurm -> torchrun settings
NPROC="${SLURM_GPUS_ON_NODE:-16}"

torchrun --standalone --nproc_per_node="${NPROC}" train.py --config configs/default.yaml

echo "Job finished at: $(date)"


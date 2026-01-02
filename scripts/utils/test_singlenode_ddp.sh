#!/bin/bash
#SBATCH --account=edu
#SBATCH --job-name=test_ddp_1node
#SBATCH --partition=short
#SBATCH --nodes=1                     # Single node test
#SBATCH --ntasks=1                    # One task; torchrun spawns processes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:05:00               # 5 minutes max
#SBATCH --gres=gpu:A6000:2            # Reduced to 2 GPUs for faster scheduling
#SBATCH --exclude=ins082
#SBATCH --output=logs/test_ddp_1node_%j.out
#SBATCH --error=logs/test_ddp_1node_%j.err

set -euo pipefail

mkdir -p logs

module load anaconda
module load cuda/12.3

eval "$(conda shell.bash hook)"
conda activate mop_clip

echo "=== TEST: Single-node DDP Setup (2 GPUs) ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Check GPU visibility
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "nvidia-smi -L:"
nvidia-smi -L || true
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
PY

# Create test script
cat > /tmp/test_ddp_init.py << 'PYEOF'
import os
import torch
import torch.distributed as dist

rank = int(os.environ.get('RANK', -1))
local_rank = int(os.environ.get('LOCAL_RANK', -1))
world_size = int(os.environ.get('WORLD_SIZE', -1))

print(f"[Rank {rank}] RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
print(f"[Rank {rank}] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Rank {rank}] CUDA device count: {torch.cuda.device_count()}")
    print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

if world_size > 1:
    try:
        # CRITICAL: Set device BEFORE initializing process group!
        # NCCL needs the correct device to be set before init_process_group()
        if torch.cuda.is_available():
            # With torchrun, all GPUs are visible and LOCAL_RANK indexes into them
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(local_rank)
            print(f"[Rank {rank}] Using device: {device} (device_count={torch.cuda.device_count()})")
        else:
            device = torch.device('cpu')
        
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        print(f"[Rank {rank}] ✓ Process group initialized!")
        
        # Test all-reduce
        tensor = torch.ones(1).to(device if torch.cuda.is_available() else 'cpu') * (rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        print(f"[Rank {rank}] All-reduce: {tensor.item()} (expected: {expected})")
        
        if abs(tensor.item() - expected) < 1e-6:
            print(f"[Rank {rank}] ✓ Communication test PASSED!")
        else:
            print(f"[Rank {rank}] ✗ Communication test FAILED!")
        
        dist.barrier()
        dist.destroy_process_group()
        print(f"[Rank {rank}] ✓ Test complete!")
    except Exception as e:
        print(f"[Rank {rank}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
PYEOF

# Use torchrun (like the working train_mop_ddp.sh script)
NPROC="${SLURM_GPUS_ON_NODE:-2}"
echo "=== Launching DDP test with torchrun (nproc_per_node=${NPROC}) ==="
torchrun --standalone --nproc_per_node="${NPROC}" /tmp/test_ddp_init.py

echo "=== TEST COMPLETE ==="
echo "Job finished at: $(date)"


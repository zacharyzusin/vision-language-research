#!/bin/bash
#SBATCH --account=edu
#SBATCH --job-name=test_ddp
#SBATCH --partition=short
#SBATCH --nodes=2                     # Use 2 nodes (8 GPUs each = 16 total)
#SBATCH --ntasks=16                   # Total tasks = 16 (one per GPU)
#SBATCH --ntasks-per-node=8           # 8 tasks per node (one per GPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb                    # Less memory for test
#SBATCH --time=00:10:00               # 10 minutes max for test
#SBATCH --gres=gpu:A6000:8
#SBATCH --exclude=ins082
#SBATCH --output=logs/test_ddp_%j.out
#SBATCH --error=logs/test_ddp_%j.err

set -euo pipefail

mkdir -p logs

module load anaconda
module load cuda/12.3

eval "$(conda shell.bash hook)"
conda activate mop_clip

echo "=== TEST: Multi-node DDP Setup ==="
echo "Job started at: $(date)"
echo "Running on nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "Hostname: $(hostname)"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"

cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

# Multi-node DDP setup
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$TOTAL_GPUS

echo "DDP Configuration:"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Number of nodes: $NNODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $TOTAL_GPUS"

# Create a simple test script that verifies DDP works
cat > /tmp/test_ddp_init.py << 'PYEOF'
import os
import torch
import torch.distributed as dist
import time

# Get environment variables
rank = int(os.environ.get('RANK', -1))
local_rank = int(os.environ.get('LOCAL_RANK', -1))
world_size = int(os.environ.get('WORLD_SIZE', -1))
master_addr = os.environ.get('MASTER_ADDR', 'unknown')
master_port = os.environ.get('MASTER_PORT', 'unknown')

print(f"[Rank {rank}] Starting DDP test...")
print(f"[Rank {rank}] RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
print(f"[Rank {rank}] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
print(f"[Rank {rank}] Hostname: {os.environ.get('SLURM_NODELIST', 'unknown')}")
print(f"[Rank {rank}] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Rank {rank}] CUDA device count: {torch.cuda.device_count()}")
    print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Initialize DDP
if world_size > 1:
    try:
        # CRITICAL: Set device BEFORE initializing process group!
        # NCCL needs the correct device to be set before init_process_group()
        if torch.cuda.is_available():
            # When using srun --gpus-per-task=1, each process sees only 1 GPU (cuda:0)
            # So we should use cuda:0, not cuda:{local_rank}
            if torch.cuda.device_count() == 1:
                device = torch.device('cuda:0')
                torch.cuda.set_device(0)
            else:
                device = torch.device(f'cuda:{local_rank}')
                torch.cuda.set_device(local_rank)
            print(f"[Rank {rank}] Using device: {device} (device_count={torch.cuda.device_count()})")
        else:
            device = torch.device('cpu')
        
        print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        print(f"[Rank {rank}] Process group initialized successfully!")
        
        # Test communication: all-reduce
        print(f"[Rank {rank}] Testing all-reduce communication...")
        tensor = torch.ones(1).to(device if torch.cuda.is_available() else 'cpu') * (rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        print(f"[Rank {rank}] All-reduce result: {tensor.item()} (expected: {expected})")
        
        if abs(tensor.item() - expected) < 1e-6:
            print(f"[Rank {rank}] ✓ DDP communication test PASSED!")
        else:
            print(f"[Rank {rank}] ✗ DDP communication test FAILED!")
        
        # Barrier to sync all processes
        print(f"[Rank {rank}] Waiting at barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Barrier passed!")
        
        # Cleanup
        dist.destroy_process_group()
        print(f"[Rank {rank}] Process group destroyed. Test complete!")
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ ERROR during DDP initialization: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print(f"[Rank {rank}] Single process mode (WORLD_SIZE=1), skipping DDP test")

print(f"[Rank {rank}] Test finished successfully!")
PYEOF

# Run the test with srun
echo ""
echo "=== Launching DDP test with srun ==="
srun --gpus-per-task=1 bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    python /tmp/test_ddp_init.py
"

echo ""
echo "=== TEST COMPLETE ==="
echo "Job finished at: $(date)"


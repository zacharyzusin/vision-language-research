# W&B Hyperparameter Sweep Setup for K Values

This directory contains scripts to run a Weights & Biases hyperparameter sweep for different K values (number of sub-prompts per class) using multi-node DDP training.

## Files Created

1. **`sweep_k_values.yaml`** - W&B sweep configuration
   - Grid search over K values: [1, 2, 4, 8, 16, 32, 64, 128]
   - Optimizes for `best_val_acc` (maximize)
   - 8 total runs

2. **`slurm/active/train_mop_ddp_sweep.sh`** - Slurm batch script for sweep agents
   - Allocates 2 nodes × 8 GPUs = 16 GPUs total
   - Runs wandb agent that executes training runs

3. **`run_sweep_training.sh`** - Wrapper script for multi-node DDP training (in project root)
   - Called by wandb agent for each sweep run
   - Sets up multi-node DDP with torchrun
   - Launches training with sweep parameters

4. **`slurm/active/launch_sweep.sh`** - Main script to initialize and submit sweep
   - Initializes W&B sweep
   - Submits Slurm jobs for sweep agents

5. **`train.py`** - Modified to support wandb.config overrides
   - Merges wandb.config into loaded config
   - Allows sweep parameters to override YAML values

## Usage

### Step 1: Make sure wandb is configured
```bash
wandb login
```

### Step 2: Run the launch script
```bash
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts
bash slurm/active/launch_sweep.sh [num_agents]
```

**Options:**
- `bash slurm/active/launch_sweep.sh` - Submit 1 agent (runs all 8 K values sequentially)
- `bash slurm/active/launch_sweep.sh 8` - Submit 8 agents (runs all 8 K values in parallel)

### Step 3: Monitor progress

**Check job status:**
```bash
squeue -u $(whoami)
```

**View logs:**
```bash
tail -f logs/sweep_*.out
```

**View sweep dashboard:**
The launch script will print the W&B sweep URL. You can also find it at:
```
https://wandb.ai/<your-entity>/inat-mop-clip/sweeps/<sweep-id>
```

## How It Works

1. **Sweep Initialization**: `slurm/active/launch_sweep.sh` creates a W&B sweep with 7 K values
2. **Agent Submission**: Slurm jobs are submitted, each running a wandb agent
3. **Run Execution**: Each agent picks up a run from the sweep:
   - Agent calls `run_sweep_training.sh` with sweep parameters
   - Wrapper script sets up multi-node DDP with torchrun
   - Training runs with the K value from the sweep
4. **Parameter Injection**: `train.py` merges `wandb.config` into the config, overriding `model.K`

## Sweep Parameters

- **K values**: 1, 2, 4, 8, 16, 32, 64, 128
- **Other hyperparameters**: From `configs/default.yaml` (batch_size=96, lr=5e-4, etc.)
- **Hardware**: 2 nodes × 8 GPUs = 16 GPUs per run
- **Effective batch size**: 96 × 16 = 1,536 per run

## Notes

- Each sweep run will train for 30 epochs (as configured in `configs/default.yaml`)
- Validation runs every 3 epochs
- Best checkpoints are saved for each K value
- All runs are logged to W&B project: `inat-mop-clip`


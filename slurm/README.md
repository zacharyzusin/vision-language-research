# Slurm Scripts Directory

This directory contains all Slurm batch scripts and launch scripts for training runs and sweeps.

## Structure

- **`active/`** - Currently active Slurm scripts
  - `train_mop_ddp_*.sh` - Slurm batch scripts (#SBATCH directives)
  - `launch_sweep_*.sh` - Scripts that initialize W&B sweeps and submit jobs
  
- **`archive/`** - Archived/old scripts (for reference)

## Usage

### Running Sweeps

From the project root, run launch scripts from this directory:

```bash
# iNaturalist sweep
bash slurm/active/launch_sweep.sh 7

# Stanford Cars sweep  
bash slurm/active/launch_sweep_stanford_cars.sh 7
```

The launch scripts will automatically reference the correct batch scripts in this directory.

## Important Notes

- Launch scripts (`launch_sweep_*.sh`) submit batch scripts using `sbatch`
- Training wrapper scripts (`run_sweep_training_*.sh`) remain in the project root because they're called by wandb agents
- Dynamically generated `launch_torchrun_*.sh` scripts are created at runtime and cleaned up automatically


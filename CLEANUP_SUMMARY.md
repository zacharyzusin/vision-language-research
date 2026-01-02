# Repository Cleanup Summary

This document summarizes the cleanup performed on the repository before pushing to GitHub.

## Changes Made

### 1. Script Organization
- Moved standalone scripts from root to `scripts/`:
  - `clip_linear_probe.py`
  - `clip_zeroshot_baseline.py`
  - `eval.py`
  - `quick_validate.py`
  - `test_model.py`
  - `validate_training_setup.py`
- Removed duplicate `download_cars_annotations.py` from root (kept version in scripts/)

### 2. Configuration Organization
- Created `configs/sweeps/` directory
- Moved all sweep configuration files (`sweep_k_values_*.yaml`) to `configs/sweeps/`
- Added `configs/sweeps/README.md` to document sweep configs

### 3. Documentation Organization
- Created `docs/` directory
- Moved all markdown documentation files to `docs/`:
  - `DATASET_DOWNLOAD.md`
  - `DIAGNOSTIC_MODE_COLLAPSE.md`
  - `K_ANALYSIS.md`
  - `KIKIBOUBA_SETUP.md`
  - `MODE_COLLAPSE_ANALYSIS.md`
  - `MODE_COLLAPSE_FIX.md`
  - `STANFORD_CARS_SETUP.md`
  - `SUBSET_TRAINING.md`
  - `SWEEP_SETUP.md`
  - `WRASSE_ABLATIONS.md`
- Added `docs/README.md` to organize documentation

### 4. Shell Script Cleanup
- Moved old/one-off shell scripts to `slurm/archive/`:
  - `launch_torchrun_sweep_*.sh` scripts
  - `run_sweep_training*.sh` scripts

### 5. .gitignore Updates
- Added `subprompt_viz_*/` pattern to ignore visualization output directories

### 6. Cleanup
- Removed stray `--help/` directory

## New Directory Structure

```
Mixture-of-Prompts/
├── configs/
│   ├── sweeps/          # W&B sweep configurations
│   └── *.yaml           # Dataset training configs
├── docs/                # All documentation
├── scripts/             # All Python scripts
│   └── utils/          # Utility scripts
├── slurm/
│   ├── active/         # Active Slurm scripts
│   └── archive/        # Archived/old scripts
├── src/                # Source code
│   ├── datasets/       # Dataset loaders
│   ├── metrics/        # Evaluation metrics
│   └── models/         # Model definitions
├── train.py            # Main training script (stays in root)
└── README.md           # Main README (stays in root)
```

## Import Paths

All scripts maintain their original import paths (e.g., `from src.models...`). 
Scripts should be run from the repository root directory for imports to work correctly.

## Next Steps

1. Review the changes
2. Test critical scripts to ensure they still work
3. Commit changes
4. Push to GitHub


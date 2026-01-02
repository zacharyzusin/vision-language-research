# Sweep Configurations

This directory contains Weights & Biases sweep configuration files for hyperparameter tuning.

## Dataset-Specific Sweeps
- `sweep_k_values_stanford_cars.yaml` - K value sweep for Stanford Cars
- `sweep_k_values_kikibouba_v1.yaml` - K value sweep for KikiBouba v1
- `sweep_k_values_kikibouba_v2.yaml` - K value sweep for KikiBouba v2
- `sweep_k_values_subset_*.yaml` - K value sweeps for various iNaturalist subsets

## Test/Development Sweeps
- `sweep_k_values_test.yaml` - Test sweep configuration
- `sweep_k_values_test_multinode.yaml` - Multi-node test sweep

## Usage
These configs are used with W&B sweeps via:
```bash
wandb sweep configs/sweeps/sweep_k_values_stanford_cars.yaml
```

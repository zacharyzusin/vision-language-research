#!/bin/bash
#SBATCH --account=edu
#SBATCH --job-name=inat2021_dl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/inat2021_download_%j.out
#SBATCH --error=logs/inat2021_download_%j.err

set -euo pipefail

mkdir -p logs
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/Mixture-of-Prompts

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "PWD: $(pwd)"

# Download + stream-extract images (no 223GB tarball stored)
export DOWNLOAD_IMAGES=1
export STREAM_EXTRACT=1

./scripts/download_inat2021.sh data/iNat2021

echo "Job finished at: $(date)"



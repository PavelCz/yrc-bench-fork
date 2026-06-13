#!/bin/bash
#SBATCH --job-name=convert_rollouts_memmap
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

DEFAULT_MANIFEST="/path/to/cluster1/data/goal-misgen/rollouts/rollouts-dummy/coinrun/gather_coinrun_exp0/rollouts_manifest_10240levels.json"
MANIFEST="${1:-$DEFAULT_MANIFEST}"

eval "$(conda shell.bash hook)"
conda activate ood-stable

mkdir -p logs/slurm

echo "Converting rollout manifest to memmap:"
echo "  ${MANIFEST}"

python scripts/convert_rollouts_to_memmap.py "${MANIFEST}"

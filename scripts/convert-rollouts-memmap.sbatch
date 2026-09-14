#!/bin/bash
#SBATCH --job-name=convert_rollouts_memmap
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

DEFAULT_MANIFEST="/nas/ucb/czempin/data/goal-misgen/rollouts/rollouts-neurips/coinrun/gather_coinrun_exp0/rollouts_manifest_10240levels.json"
MANIFEST="${1:-$DEFAULT_MANIFEST}"

CONDA_BASE="${CONDA_BASE:-/nas/ucb/czempin/anaconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found (CONDA_BASE=$CONDA_BASE)" >&2
    exit 1
fi
echo "Using conda env: ood-stable"
conda activate ood-stable

mkdir -p logs/slurm

echo "Converting rollout manifest to memmap:"
echo "  ${MANIFEST}"

python scripts/convert_rollouts_to_memmap.py "${MANIFEST}"

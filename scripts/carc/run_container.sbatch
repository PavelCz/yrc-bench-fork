#!/bin/bash

#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=yrc-container
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load apptainer

REPO_DIR="${REPO_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$PWD}}"
if command -v git >/dev/null 2>&1; then
    GIT_ROOT="$(git -C "${REPO_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${GIT_ROOT}" ]]; then
        REPO_DIR="${GIT_ROOT}"
    fi
fi

SIF_NAME="${YRC_SIF_NAME:-yrc-bench-procgen.sif}"
SIF_PATH="${YRC_SIF_PATH:-${REPO_DIR}/${SIF_NAME}}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Apptainer image not found: ${SIF_PATH}"
    echo "Build it with: sbatch scripts/carc/build_apptainer.sbatch"
    exit 1
fi

for path in "YRC" "configs"; do
    if [[ ! -e "${REPO_DIR}/${path}" ]]; then
        echo "Required runtime path is missing: ${REPO_DIR}/${path}"
        echo "If this came from a submodule, run: git submodule update --init --recursive"
        exit 1
    fi
done

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/.container_cache"

APPTAINER_ARGS=(
    exec
    --nv
    --bind "${REPO_DIR}:/workspace"
    --bind "/project2/biyik_1165:/project2/biyik_1165"
    --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=/workspace"
    --env "MPLCONFIGDIR=/workspace/.container_cache/matplotlib"
    --env "XDG_CACHE_HOME=/workspace/.container_cache/xdg"
    --env "WANDB_CACHE_DIR=/workspace/.container_cache/wandb"
    --pwd /workspace
)

if [[ -n "${YRC_EXTRA_BINDS:-}" ]]; then
    IFS=',' read -ra EXTRA_BINDS <<< "${YRC_EXTRA_BINDS}"
    for bind_path in "${EXTRA_BINDS[@]}"; do
        APPTAINER_ARGS+=(--bind "${bind_path}")
    done
fi

if [[ $# -eq 0 ]]; then
    CMD=(bash)
else
    CMD=("$@")
fi

apptainer "${APPTAINER_ARGS[@]}" "${SIF_PATH}" "${CMD[@]}"

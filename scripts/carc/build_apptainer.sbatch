#!/bin/bash

#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=yrc-build-apptainer
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

if [[ ! -f "${REPO_DIR}/docker/Dockerfile" ]]; then
    echo "Could not locate repo root at '${REPO_DIR}'. Submit from the repo root or set REPO_DIR_OVERRIDE."
    exit 1
fi

required_paths=(
    "requirements.txt"
    "docker/Dockerfile"
    "lib/acs/pyproject.toml"
    "lib/procgen/setup.py"
    "lib/procgen/procgen/src/game-registry.cpp"
    "lib/pytorch_vae/models/__init__.py"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "${REPO_DIR}/${path}" ]]; then
        echo "Required source path is missing: ${REPO_DIR}/${path}"
        echo "If this came from a submodule, run:"
        echo "  git submodule update --init --recursive"
        exit 1
    fi
done

SIF_NAME="${1:-${YRC_SIF_NAME:-yrc-bench-procgen.sif}}"
SIF_PATH="${REPO_DIR}/${SIF_NAME}"

mkdir -p "${REPO_DIR}/logs"

DEFAULT_APPTAINER_ROOT="/project2/biyik_1165/${USER}/tmp/apptainer"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${DEFAULT_APPTAINER_ROOT}/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${DEFAULT_APPTAINER_ROOT}/cache}"
export GOMAXPROCS="${GOMAXPROCS:-4}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

if command -v spython >/dev/null 2>&1; then
    SPYTHON_CMD=(spython)
elif python -m spython --help >/dev/null 2>&1; then
    SPYTHON_CMD=(python -m spython)
else
    echo "spython is not available. Install it on CARC, for example: pipx install spython"
    exit 1
fi

cd "${REPO_DIR}"
"${SPYTHON_CMD[@]}" recipe docker/Dockerfile > apptainer.def
apptainer build "${SIF_PATH}" apptainer.def

echo "Built ${SIF_PATH}"

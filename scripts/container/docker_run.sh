#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"

IMAGE="${YRC_DOCKER_IMAGE:-yrc-bench-procgen:latest}"

if [[ $# -eq 0 ]]; then
    CMD=(bash)
else
    CMD=("$@")
fi

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_ARGS=(--gpus all)
fi

mkdir -p "${REPO_DIR}/.container_cache"

docker run --rm -it \
    "${GPU_ARGS[@]}" \
    --shm-size="${YRC_DOCKER_SHM_SIZE:-16g}" \
    -v "${REPO_DIR}:/workspace" \
    -w /workspace \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/workspace \
    -e MPLCONFIGDIR=/workspace/.container_cache/matplotlib \
    -e XDG_CACHE_HOME=/workspace/.container_cache/xdg \
    -e WANDB_CACHE_DIR=/workspace/.container_cache/wandb \
    "${IMAGE}" \
    "${CMD[@]}"

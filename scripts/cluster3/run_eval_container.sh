#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"

CONTAINER_IMAGE="${YRC_SIF_PATH:-${REPO_DIR}/yrc-bench-procgen.sif}"

python "${REPO_DIR}/scripts/run_eval.py" \
    --server cluster3 \
    --use-container \
    --container-image "${CONTAINER_IMAGE}" \
    "$@"

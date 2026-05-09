#!/bin/bash

#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=gpu-overlap-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

NUM_STEPS="${NUM_STEPS:-4}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

mkdir -p logs

echo "job=${SLURM_JOB_ID:-unset}"
echo "node=${SLURMD_NODENAME:-unset}"
echo "num_steps=${NUM_STEPS}"
echo "sleep_seconds=${SLEEP_SECONDS}"
echo "submit_time=$(date +%F_%T)"
echo

pids=()
for step_id in $(seq 0 "$((NUM_STEPS - 1))"); do
    srun --overlap \
        --ntasks=1 \
        --cpus-per-task=1 \
        --gpus-per-task=1 \
        bash -lc \
            "echo step=${step_id} start \$(date +%T) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}; sleep ${SLEEP_SECONDS}; echo step=${step_id} end \$(date +%T)" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo
echo "finish_time=$(date +%F_%T)"
exit "${status}"

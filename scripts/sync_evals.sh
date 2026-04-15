#!/usr/bin/env bash
set -uo pipefail  # Remove 'e' flag to allow continuing after errors

SRC_BASE="rnn:/nas/ucb/czempin/data/goal-misgen/experiments/evals"
DST_BASE="/home/pavel/data/goal-misgen/icml-evals"

PREFIX="${1:-icml04}"
ENVS=("maze" "coinrun")
EXPS=("exp0" "exp1" "exp2" "exp3")

failed_syncs=0

for env in "${ENVS[@]}"; do
  for exp in "${EXPS[@]}"; do
    src_dir="${SRC_BASE}/${PREFIX}_${env}_${exp}"
    dst_dir="${DST_BASE}/"

    echo "Syncing ${src_dir} -> ${dst_dir}"
    if rsync -av --progress "${src_dir}" "${dst_dir}"; then
      echo "✓ Successfully synced ${env}_${exp}"
    else
      echo "✗ Failed to sync ${env}_${exp} (continuing...)"
      ((failed_syncs++))
    fi
  done
done

if [ $failed_syncs -gt 0 ]; then
  echo "Warning: $failed_syncs sync(s) failed"
  exit 1
else
  echo "All syncs completed successfully"
fi


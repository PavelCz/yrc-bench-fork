#!/usr/bin/env bash
set -uo pipefail  # Remove 'e' flag to allow continuing after errors

SRC_BASE="rnn:/nas/ucb/czempin/data/goal-misgen/experiments/evals"
DST_BASE="/home/pavel/data/goal-misgen/icml-evals"

PREFIX="${1:-icml04}"
ENVS=("maze" "coinrun")
EXPS=("exp0" "exp1" "exp2" "exp3")

# Parse "host:path" into host and path for remote existence checks
SRC_HOST="${SRC_BASE%%:*}"
SRC_PATH="${SRC_BASE#*:}"

failed_syncs=0
missing_syncs=0

for env in "${ENVS[@]}"; do
  for exp in "${EXPS[@]}"; do
    name="${PREFIX}_${env}_${exp}"
    src_dir="${SRC_BASE}/${name}"
    dst_dir="${DST_BASE}/"

    if ! ssh "${SRC_HOST}" "test -d ${SRC_PATH}/${name}" 2>/dev/null; then
      echo "– Skipping ${name}: not present on ${SRC_HOST}"
      ((missing_syncs++))
      continue
    fi

    echo "Syncing ${src_dir} -> ${dst_dir}"
    if rsync -av --progress "${src_dir}" "${dst_dir}"; then
      echo "✓ Successfully synced ${name}"
    else
      echo "✗ Failed to sync ${name} (continuing...)"
      ((failed_syncs++))
    fi
  done
done

if [ $missing_syncs -gt 0 ]; then
  echo "Note: $missing_syncs source dir(s) did not exist on ${SRC_HOST} and were skipped"
fi
if [ $failed_syncs -gt 0 ]; then
  echo "Warning: $failed_syncs sync(s) failed"
  exit 1
else
  echo "All available syncs completed successfully"
fi


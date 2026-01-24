#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="rnn:/nas/ucb/czempin/data/goal-misgen/experiments/evals"
DST_BASE="/home/pavel/data/goal-misgen/icml-evals"

ENVS=("maze" "coinrun")
EXPS=("exp0" "exp1" "exp2")

for env in "${ENVS[@]}"; do
  for exp in "${EXPS[@]}"; do
    src_dir="${SRC_BASE}/icml-test_${env}_${exp}"
    dst_dir="${DST_BASE}/"

    echo "Syncing ${src_dir} -> ${dst_dir}"
    rsync -av --progress "${src_dir}" "${dst_dir}"
  done
done


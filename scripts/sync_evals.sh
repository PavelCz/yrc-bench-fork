#!/usr/bin/env bash
set -uo pipefail  # Remove 'e' flag to allow continuing after errors

SRC_BASE="rnn:/nas/ucb/czempin/data/goal-misgen/experiments/evals"
DST_BASE="/home/pavel/data/goal-misgen/icml-evals"

usage() {
  echo "Usage: $0 [--with-videos] [prefix]"
  echo
  echo "Sync eval directories from ${SRC_BASE} to ${DST_BASE}."
  echo "Videos and images in the videos folder are excluded by default."
}

SYNC_VIDEOS=0
PREFIX="icml04"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-videos)
      SYNC_VIDEOS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      PREFIX="$1"
      shift
      ;;
  esac
done

ENVS=("maze" "coinrun")
EXPS=("exp0" "exp1" "exp2" "exp3")

# Parse "host:path" into host and path for remote existence checks
SRC_HOST="${SRC_BASE%%:*}"
SRC_PATH="${SRC_BASE#*:}"

failed_syncs=0
missing_syncs=0
RSYNC_ARGS=(-av --progress)

if [[ "${SYNC_VIDEOS}" -eq 0 ]]; then
  RSYNC_ARGS+=(
    --exclude='videos/'
    --exclude='videos/**'
    --exclude='*.mp4'
    --exclude='*.webm'
    --exclude='*.gif'
    --exclude='*.png'
    --exclude='*.jpg'
    --exclude='*.jpeg'
  )
fi

for env in "${ENVS[@]}"; do
  for exp in "${EXPS[@]}"; do
    name="${PREFIX}_${env}_${exp}"
    src_dir="${SRC_BASE}/${name}"
    dst_dir="${DST_BASE}/"

    if ! ssh "${SRC_HOST}" "test -d ${SRC_PATH}/${name}" 2>/dev/null; then
      echo "– Skipping ${name}: not present on ${SRC_HOST}"
      missing_syncs=$((missing_syncs + 1))
      continue
    fi

    echo "Syncing ${src_dir} -> ${dst_dir}"
    if rsync "${RSYNC_ARGS[@]}" "${src_dir}" "${dst_dir}"; then
      echo "✓ Successfully synced ${name}"
    else
      echo "✗ Failed to sync ${name} (continuing...)"
      failed_syncs=$((failed_syncs + 1))
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


#!/usr/bin/env bash
set -uo pipefail  # Remove 'e' flag to allow continuing after errors

SRC_BASE="cluster1:/path/to/cluster1/data/goal-misgen/experiments/evals"
DST_BASE="/home/user/data/goal-misgen/dummy-evals"

usage() {
  echo "Usage: $0 [--with-videos] [prefix]"
  echo
  echo "Sync eval directories from ${SRC_BASE} to ${DST_BASE}."
  echo "Includes both standard eval dirs (<prefix>_<env>_expN) and"
  echo "policy-eval dirs (<prefix>_<env>_<agent>_expN)."
  echo "Also includes robust maze eval dirs (<prefix>_robust{200,400}_<env>_expN)."
  echo "Also includes robust policy-eval dirs (<prefix>_robust{200,400}_<env>_strong_expN)."
  echo "Videos and images in the videos folder are excluded by default."
}

SYNC_VIDEOS=0
PREFIX="dummy04"

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

ENVS=("maze" "coinrun" "coinrun_proxy_fail" "maze_proxy_fail")
ROBUST_ENVS=("maze" "maze_proxy_fail")
ROBUST_VARIANTS=("robust200" "robust400")
EXPS=("exp0" "exp1" "exp2" "exp3")
AGENTS=("sim" "weak" "strong")

# Parse "host:path" into host and path for remote existence checks
SRC_HOST="${SRC_BASE%%:*}"
SRC_PATH="${SRC_BASE#*:}"

failed_syncs=0
missing_syncs=0
available_syncs=0
SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
RSYNC_ARGS=(-av --progress -e "ssh ${SSH_OPTS[*]}")

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

echo "Listing remote directories under ${SRC_BASE}..."
if ! REMOTE_LISTING="$(ssh "${SSH_OPTS[@]}" "${SRC_HOST}" "ls -1 ${SRC_PATH}" 2>/dev/null)"; then
  echo "✗ Failed to list remote directory ${SRC_BASE}" >&2
  exit 1
fi

remote_has() {
  local name="$1"
  grep -Fxq "${name}" <<<"${REMOTE_LISTING}"
}

sync_name() {
  local name="$1"
  local src_dir="${SRC_BASE}/${name}"
  local dst_dir="${DST_BASE}/"

  if ! remote_has "${name}"; then
    echo "– Skipping ${name}: not present on ${SRC_HOST}"
    missing_syncs=$((missing_syncs + 1))
    return
  fi

  available_syncs=$((available_syncs + 1))
  echo "Syncing ${src_dir} -> ${dst_dir}"
  if rsync "${RSYNC_ARGS[@]}" "${src_dir}" "${dst_dir}"; then
    echo "✓ Successfully synced ${name}"
  else
    echo "✗ Failed to sync ${name} (continuing...)"
    failed_syncs=$((failed_syncs + 1))
  fi
}

for env in "${ENVS[@]}"; do
  for exp in "${EXPS[@]}"; do
    # Standard AFHP eval directories
    sync_name "${PREFIX}_${env}_${exp}"

    # Agent-specific policy eval directories
    for agent in "${AGENTS[@]}"; do
      sync_name "${PREFIX}_${env}_${agent}_${exp}"
    done
  done
done

for robust_variant in "${ROBUST_VARIANTS[@]}"; do
  for env in "${ROBUST_ENVS[@]}"; do
    for exp in "${EXPS[@]}"; do
      # Robust AFHP eval directories
      sync_name "${PREFIX}_${robust_variant}_${env}_${exp}"

      # Robust strong-policy eval directories
      sync_name "${PREFIX}_${robust_variant}_${env}_strong_${exp}"
    done
  done
done

if [ $missing_syncs -gt 0 ]; then
  echo "Note: $missing_syncs source dir(s) did not exist on ${SRC_HOST} and were skipped"
fi
if [ $available_syncs -eq 0 ]; then
  echo "Error: no source dirs matched prefix '${PREFIX}' on ${SRC_HOST}; nothing was available to sync"
  exit 1
fi
if [ $failed_syncs -gt 0 ]; then
  echo "Warning: $failed_syncs sync(s) failed"
  exit 1
else
  echo "All available syncs completed successfully"
fi

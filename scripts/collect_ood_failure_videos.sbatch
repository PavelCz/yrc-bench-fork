#!/bin/bash
#
# Collect weak-policy Heist videos of ground-truth OOD timeouts.
#
# Usage on rnn:
#   mkdir -p /nas/ucb/czempin/data/goal-misgen/slurm-logs/default/tmlr-heist02-ood-timeout-videos
#   sbatch scripts/collect_ood_failure_videos.sbatch
#   EXP_ID=1 VIDEO_EPISODES=8 sbatch scripts/collect_ood_failure_videos.sbatch
#
# SLURM spools a copy of this file, so BASH_SOURCE cannot locate the repo.
# Set PROJECT_ROOT if you are submitting from a worktree.
#
#SBATCH --job-name=heist_ood_timeout_videos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=0-08:00:00
#SBATCH --qos=default
#SBATCH --output=/nas/ucb/czempin/data/goal-misgen/slurm-logs/default/tmlr-heist02-ood-timeout-videos/%x_%j.out
#SBATCH --error=/nas/ucb/czempin/data/goal-misgen/slurm-logs/default/tmlr-heist02-ood-timeout-videos/%x_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-ood-stable}"
CONDA_BASE="${CONDA_BASE:-/nas/ucb/czempin/anaconda3}"
PROJECT_ROOT="${PROJECT_ROOT:-/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/nas/ucb/czempin/data/goal-misgen/policy/icml}"
SEEDS_BASE="${SEEDS_BASE:-/nas/ucb/czempin/data/goal-misgen/seeds/icml}"
CONFIG="${CONFIG:-configs/eval/heist/max_prob.yaml}"
PREFIX="${PREFIX:-tmlr-heist02-ood-timeout-videos}"
EXP_ID="${EXP_ID:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-256}"
VIDEO_EPISODES="${VIDEO_EPISODES:-16}"
GREEDY="${GREEDY:-false}"
MODEL_FILE="${MODEL_FILE:-}"

JOB_NAME="heist_ood_timeout_videos_exp${EXP_ID}"
EXPERIMENT_GROUP="${PREFIX}_heist_weak_exp${EXP_ID}"
SEEDS_FILE="${SEEDS_FILE:-${SEEDS_BASE}/${EXP_ID}.json}"

if [[ ! -f "${PROJECT_ROOT}/collect_ood_failure_videos.py" ]]; then
    echo "ERROR: collect_ood_failure_videos.py not found under ${PROJECT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SEEDS_FILE}" ]]; then
    echo "ERROR: level seeds file not found: ${SEEDS_FILE}" >&2
    exit 1
fi

mkdir -p /nas/ucb/czempin/data/goal-misgen/slurm-logs/default/tmlr-heist02-ood-timeout-videos

# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

if [[ -z "${MODEL_FILE}" ]]; then
    MODEL_FILE="$(
        python -c "
import sys
sys.path.insert(0, 'scripts')
from common import get_checkpoints
print(get_checkpoints('heist', ${EXP_ID}, '${CHECKPOINT_BASE}')['weak'])
"
    )"
fi
if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "ERROR: weak checkpoint not found: ${MODEL_FILE}" >&2
    exit 1
fi

echo "job=${SLURM_JOB_ID:-unset} node=${SLURMD_NODENAME:-unset}"
echo "exp_id=${EXP_ID} prefix=${PREFIX}"
echo "model_file=${MODEL_FILE}"
echo "seeds_file=${SEEDS_FILE}"
echo "video_episodes=${VIDEO_EPISODES} num_rollouts=${NUM_ROLLOUTS} greedy=${GREEDY}"
echo "start_time=$(date +%F_%T)"
echo

export PYTHONUNBUFFERED=1
python collect_ood_failure_videos.py \
    -c "${CONFIG}" \
    -n "${JOB_NAME}" \
    -experiment_group "${EXPERIMENT_GROUP}" \
    -en heist_afh \
    --model_file "${MODEL_FILE}" \
    -level_seeds_file "${SEEDS_FILE}" \
    -eval_split "${EVAL_SPLIT}" \
    -num_rollouts "${NUM_ROLLOUTS}" \
    -video_episodes_to_collect "${VIDEO_EPISODES}" \
    -video_logging_mode folder \
    -wandb_mode disabled \
    -greedy "${GREEDY}"

echo
echo "finish_time=$(date +%F_%T)"

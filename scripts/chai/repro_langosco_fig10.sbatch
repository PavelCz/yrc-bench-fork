#!/bin/bash
#
# Reproduce Langosco et al. Fig 10 (heist "keys and chests") — see issue #23.
#
# This mirrors their training script
# lib/train-procgen-pytorch/experiments/scripts/train-keys-and-chests.sh
# (byte-identical to upstream jbkjr/train-procgen-pytorch @ 2906e6f), through
# our train.py. Our PPO `optimize`/`predict` are AST-identical to upstream and
# our ProcgenEnv construction matches modulo kwargs that are no-ops at their
# defaults.
#
# IMPORTANT — script knobs ≠ paper description / our coinrun–maze baseline:
#   train-keys-and-chests.sh sets key_penalty=3 (−0.3 per key) and
#   --param_name A100 (80M steps). That is not what the paper's keys-and-chests
#   writeup describes as the mainline setup, and it is not what
#   scripts/train_policies.sh uses for coinrun/maze/heist (param_name=paper,
#   key_penalty unset → 0, 200M). This sbatch intentionally tracks the SCRIPT;
#   treat key_penalty=3 as script-specific (plausibly an ablation). See issue #23.
#
# Their training loop IS the eval: it steps the OOD env in lockstep with
# training and logs both to the same row of {logdir}/log-append.csv:
#
#     heist_aisc_many_chests  ->  Rewards/mean_episodes           (ID)
#     heist_aisc_many_keys    ->  [Valid] Rewards/mean_episodes   (OOD)
#
# Returns are logged raw, not normalized: storage.fetch_log_data() prefers
# info['env_reward'] (pre-VecNormalize) and only falls back to the normalized
# tensor if absent. Same on both sides, so the y-axis is comparable to ours.
#
# DELIBERATE DEVIATIONS from train-keys-and-chests.sh:
#   --seed                    pinned (their default is random.randint(0, 9999))
#   --logdir_base             ours; routes checkpoints off the repo
#   --log_interval            set to one PPO iteration, see LOG_INTERVAL below
#   --num_validation_episodes 40, see the note on the deque below
#   --use_wandb               off by default; log-append.csv has everything
#
# Everything else is left unset so it takes the same default their script got:
# no --num_levels (=0, unbounded), no --level_seeds_file (stock procgen level
# sampling, and recreate_eval_env stays False), no --random_percent (unused by
# the heist_aisc_* envs), backgrounds on.
#
# Usage:
#   sbatch scripts/chai/repro_langosco_fig10.sbatch
#   SEED=2 EXP_NAME=langosco-fig10-s2 sbatch scripts/chai/repro_langosco_fig10.sbatch
#   KEY_PENALTY=0 EXP_NAME=langosco-fig10-nopenalty sbatch ...   # the ablation
#   # Resume 80M -> ~200M cumulative (self.t resets; NUM_TIMESTEPS is additional):
#   MODEL_FILE=auto NUM_TIMESTEPS=120000000 sbatch --time=2-00:00:00 \
#     scripts/chai/repro_langosco_fig10.sbatch
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=langosco-fig10
#SBATCH --output=/nas/ucb/czempin/data/goal-misgen/logs/train_policies/%x_%j.out

set -euo pipefail

# --- knobs -------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-ood-stable}"
CONDA_BASE="${CONDA_BASE:-/nas/ucb/czempin/anaconda3}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/nas/ucb/czempin/data/goal-misgen/policy/langosco-repro}"
EXP_NAME="${EXP_NAME:-langosco-fig10}"
SEED="${SEED:-1111}"
# Resume from the latest checkpoint under EXP_NAME (reuses that run's logdir).
# Note: PPO's self.t resets to 0 on load, so NUM_TIMESTEPS is *additional*
# steps — use 120000000 to reach ~200M cumulative after an 80M run.
MODEL_FILE="${MODEL_FILE:-}"

# --- theirs, verbatim from train-keys-and-chests.sh --------------------------
ENV_NAME="heist_aisc_many_chests"
VAL_ENV_NAME="heist_aisc_many_keys"
KEY_PENALTY="${KEY_PENALTY:-3}"
PARAM_NAME="${PARAM_NAME:-A100}"
DISTRIBUTION_MODE="hard"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-80000000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-4}"
NUM_THREADS="${NUM_THREADS:-32}"

# --- derived -----------------------------------------------------------------
# Must be absolute: SLURM runs a spooled copy of this script from /var/spool/slurm,
# so BASH_SOURCE cannot be used to locate the repo from inside the job. Override
# PROJECT_ROOT to submit a worktree's code instead.
PROJECT_ROOT="${PROJECT_ROOT:-/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork}"
TRAIN_DIR="${PROJECT_ROOT}/lib/train-procgen-pytorch"

if [ ! -f "${TRAIN_DIR}/train.py" ]; then
    echo "ERROR: no train.py under ${TRAIN_DIR} — set PROJECT_ROOT to the repo root." >&2
    exit 1
fi

# Upstream validates every PPO iteration, ungated. Ours gates on --log_interval,
# so set it to exactly one iteration's timesteps (n_steps * n_envs) to match the
# cadence. Read from the hyperparameter set rather than hardcoded, so changing
# PARAM_NAME stays consistent.
LOG_INTERVAL="$(python3 -c "
import yaml
h = yaml.safe_load(open('${TRAIN_DIR}/hyperparams/procgen/config.yml'))['${PARAM_NAME}']
print(h['n_steps'] * h['n_envs'])
")"

# The validation buffer is a deque(maxlen=40) — episodes beyond the last 40 are
# discarded before the mean is taken, so a larger value costs env steps and
# changes nothing. 40 is also upstream's effective sample size, which is what we
# want here. NOTE: this cap applies to our existing training runs too; their
# val curves are 40-episode means regardless of --num_validation_episodes.
# The precise per-checkpoint numbers come from eval_policy.py in step 3, not
# from this curve.
NUM_VALIDATION_EPISODES="${NUM_VALIDATION_EPISODES:-40}"

echo "job=${SLURM_JOB_ID:-unset} node=${SLURMD_NODENAME:-unset}"
echo "exp_name=${EXP_NAME} seed=${SEED} key_penalty=${KEY_PENALTY}"
echo "param_name=${PARAM_NAME} log_interval=${LOG_INTERVAL}"
echo "model_file=${MODEL_FILE:-<new run>}"
echo "num_timesteps=${NUM_TIMESTEPS}"
echo "logdir=${CHECKPOINT_BASE}/${ENV_NAME}/${EXP_NAME}/<run_name>"
echo "start_time=$(date +%F_%T)"
echo

# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"
cd "${TRAIN_DIR}"

MODEL_FILE_ARGS=()
if [ -n "${MODEL_FILE}" ]; then
    MODEL_FILE_ARGS=(--model_file "${MODEL_FILE}")
fi

conda run --no-capture-output -n "${CONDA_ENV}" python train.py \
    --exp_name "${EXP_NAME}" \
    --env_name "${ENV_NAME}" \
    --val_env_name "${VAL_ENV_NAME}" \
    --key_penalty "${KEY_PENALTY}" \
    --param_name "${PARAM_NAME}" \
    --distribution_mode "${DISTRIBUTION_MODE}" \
    --num_timesteps "${NUM_TIMESTEPS}" \
    --num_checkpoints "${NUM_CHECKPOINTS}" \
    --num_threads "${NUM_THREADS}" \
    --log_interval "${LOG_INTERVAL}" \
    --num_validation_episodes "${NUM_VALIDATION_EPISODES}" \
    --seed "${SEED}" \
    --logdir_base "${CHECKPOINT_BASE}" \
    "${MODEL_FILE_ARGS[@]}"

echo
echo "finish_time=$(date +%F_%T)"
echo "Checkpoints (model_{t}.pth, ~20/40/60/80M):"
find "${CHECKPOINT_BASE}/${ENV_NAME}/${EXP_NAME}" -name 'model_*.pth' 2>/dev/null | sort

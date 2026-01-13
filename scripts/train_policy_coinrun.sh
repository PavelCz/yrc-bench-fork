#!/bin/bash

# Get script directory and project paths (resolve symlinks for slurm compatibility)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$(dirname "$SCRIPT_DIR")")"
TRAIN_SCRIPT="${PROJECT_ROOT}/lib/train-procgen-pytorch/train.py"

# Configuration
#
# Different experiment configs correspond to different "seeds." Experiment 0
# reproduces the settings from the original paper by Langosco et al.
#
# EXPERIMENT_ID | SEED        | LEVEL_SEEDS_FILE | TRAIN_MODE | NUM_LEVELS
# --------------|-------------|------------------|------------|------------
# 0             | 6033        | 0.json           | fallback   | 100000
# 1-4           | same as ID  | {ID}.json        | random     | (not set)
#
CONDA_ENV="ood-stable"
EXP_PREFIX="icml2"
EXPERIMENT_ID=0
LEVEL_SEEDS_FOLDER="/nas/ucb/czempin/data/goal-misgen/seeds/icml"
LOG_DIR="/nas/ucb/czempin/data/goal-misgen/logs/train_policy_coinrun"
RANDOM_PERCENTS=(0 50 100)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set experiment-specific parameters
if [ "$EXPERIMENT_ID" -eq 0 ]; then
    SEED=6033
    LEVEL_SEEDS_FILE="0.json"
    TRAIN_MODE="fallback"
    NUM_LEVELS=100000
else
    SEED="$EXPERIMENT_ID"
    LEVEL_SEEDS_FILE="${EXPERIMENT_ID}.json"
    TRAIN_MODE="random"
    NUM_LEVELS=""
fi

for random_percent in "${RANDOM_PERCENTS[@]}"; do
    exp_name="${EXP_PREFIX}_${random_percent}p"

    # Build optional arguments
    EXTRA_ARGS=""
    if [ -n "$NUM_LEVELS" ]; then
        EXTRA_ARGS="--num_levels $NUM_LEVELS"
    fi

    sbatch --qos=default \
        --gres=gpu:1 \
        --time=3-00:00:00 \
        --mem=128G \
        --job-name="$exp_name" \
        --output="${LOG_DIR}/${exp_name}.out" \
        --wrap="conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
            --level_seeds_file ${LEVEL_SEEDS_FOLDER}/${LEVEL_SEEDS_FILE} \
            --train_mode $TRAIN_MODE \
            --eval_mode sequential \
            --exp_name $exp_name \
            --env_name coinrun \
            --val_env_name coinrun \
            --random_percent $random_percent \
            --random_percent_val 50 \
            --distribution_mode hard \
            --param_name paper \
            --num_timesteps 200000000 \
            --log_interval 4000000 \
            --num_checkpoints 10 \
            --num_threads 4 \
            --seed $SEED \
            $EXTRA_ARGS"
done

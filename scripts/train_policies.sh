#!/bin/bash

# Get script directory and project paths (resolve symlinks for slurm compatibility)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$(dirname "$SCRIPT_DIR")")"
TRAIN_DIR="${PROJECT_ROOT}/lib/train-procgen-pytorch"

# Default configuration
CONDA_ENV="ood-stable"
EXP_PREFIX="icml2"
LEVEL_SEEDS_FOLDER="/nas/ucb/czempin/data/goal-misgen/seeds/icml"
LOG_DIR="/nas/ucb/czempin/data/goal-misgen/logs/train_policies"
RANDOM_PERCENTS=(0 50 100)

# Usage function
usage() {
    cat << EOF
Usage: $(basename "$0") -e ENV_TYPE -x EXPERIMENT_ID

Train policies for goal misgeneralization experiments.

Required arguments:
    -e, --env ENV_TYPE        Environment type: "coinrun" or "maze"
    -x, --experiment ID       Experiment ID: 0-4

Optional arguments:
    -h, --help                Show this help message

Experiment configurations:
    EXPERIMENT_ID | SEED              | LEVEL_SEEDS_FILE | TRAIN_MODE | NUM_LEVELS
    --------------|-------------------|------------------|------------|------------
    0             | 6033/1080 (env)   | 0.json           | fallback   | 100000
    1-4           | same as ID        | {ID}.json        | random     | (not set)

Examples:
    $(basename "$0") -e coinrun -x 0
    $(basename "$0") --env maze --experiment 2
EOF
    exit 1
}

# Parse command line arguments
ENV_TYPE=""
EXPERIMENT_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENV_TYPE="$2"
            shift 2
            ;;
        -x|--experiment)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$ENV_TYPE" ]; then
    echo "Error: ENV_TYPE is required (-e or --env)"
    usage
fi

if [ -z "$EXPERIMENT_ID" ]; then
    echo "Error: EXPERIMENT_ID is required (-x or --experiment)"
    usage
fi

# Validate ENV_TYPE
if [ "$ENV_TYPE" != "coinrun" ] && [ "$ENV_TYPE" != "maze" ]; then
    echo "Error: ENV_TYPE must be 'coinrun' or 'maze', got '$ENV_TYPE'"
    exit 1
fi

# Validate EXPERIMENT_ID is a number 0-4
if ! [[ "$EXPERIMENT_ID" =~ ^[0-4]$ ]]; then
    echo "Error: EXPERIMENT_ID must be an integer between 0 and 4, got '$EXPERIMENT_ID'"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set experiment-specific parameters
if [ "$EXPERIMENT_ID" -eq 0 ]; then
    LEVEL_SEEDS_FILE="0.json"
    TRAIN_MODE="fallback"
    NUM_LEVELS=100000
else
    LEVEL_SEEDS_FILE="${EXPERIMENT_ID}.json"
    TRAIN_MODE="random"
    NUM_LEVELS=""
fi

# Set environment-specific parameters
if [ "$ENV_TYPE" = "coinrun" ]; then
    ENV_NAME="coinrun"
    VAL_ENV_NAME="coinrun"
    DEFAULT_SEED=6033
elif [ "$ENV_TYPE" = "maze" ]; then
    ENV_NAME="maze_afh"
    VAL_ENV_NAME="maze_afh"
    DEFAULT_SEED=1080
fi

# Set seed based on experiment ID
if [ "$EXPERIMENT_ID" -eq 0 ]; then
    SEED="$DEFAULT_SEED"
else
    SEED="$EXPERIMENT_ID"
fi

echo "Starting training with:"
echo "  ENV_TYPE:      $ENV_TYPE"
echo "  EXPERIMENT_ID: $EXPERIMENT_ID"
echo "  SEED:          $SEED"
echo "  TRAIN_MODE:    $TRAIN_MODE"
echo ""

for random_percent in "${RANDOM_PERCENTS[@]}"; do
    exp_name="${EXP_PREFIX}_${ENV_TYPE}_exp${EXPERIMENT_ID}_${random_percent}p"

    # Build optional arguments
    EXTRA_ARGS=""
    if [ -n "$NUM_LEVELS" ]; then
        EXTRA_ARGS="--num_levels $NUM_LEVELS"
    fi

    echo "Submitting job: $exp_name"

    sbatch --qos=default \
        --gres=gpu:1 \
        --time=3-00:00:00 \
        --mem=128G \
        --job-name="$exp_name" \
        --output="${LOG_DIR}/${exp_name}.out" \
        --wrap="cd $TRAIN_DIR && conda run -n $CONDA_ENV python train.py \
            --level_seeds_file ${LEVEL_SEEDS_FOLDER}/${LEVEL_SEEDS_FILE} \
            --train_mode $TRAIN_MODE \
            --eval_mode sequential \
            --exp_name $exp_name \
            --env_name $ENV_NAME \
            --val_env_name $VAL_ENV_NAME \
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

echo ""
echo "All jobs submitted. Check logs in: $LOG_DIR"

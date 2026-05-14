#!/bin/bash

# Get script directory and project paths (resolve symlinks for slurm compatibility)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$(dirname "$SCRIPT_DIR")")"
TRAIN_DIR="${PROJECT_ROOT}/lib/train-procgen-pytorch"

# Default configuration
CONDA_ENV="ood"
CONDA_BASE="/nas/ucb/czempin/anaconda3"
EXP_PREFIX="icml2"
LEVEL_SEEDS_FOLDER="/nas/ucb/czempin/data/goal-misgen/seeds/icml"
LOG_DIR="/nas/ucb/czempin/data/goal-misgen/logs/train_policies"
CHECKPOINT_BASE="/nas/ucb/czempin/data/goal-misgen/policy/icml"
RANDOM_PERCENTS=(0 50 100)

# Usage function
usage() {
    cat << EOF
Usage: $(basename "$0") -e ENV_TYPE -x EXPERIMENT_ID

Train policies for goal misgeneralization experiments.

Required arguments:
    -e, --env ENV_TYPE        Environment type: "coinrun", "maze", or "heist"
    -x, --experiment ID       Experiment ID: 0-4

Optional arguments:
    -h, --help                Show this help message
    --randomize-agent-start   MAZE only: train with randomized initial agent cells
    --random-percent PERCENT  Train only this random_percent value (default: 0 50 100)
    --num-timesteps N         Training timesteps per job (default: 200000000)
    --days N                  SLURM wall-time days per job (default: 3)

Experiment configurations:
    EXPERIMENT_ID | SEED                   | LEVEL_SEEDS_FILE | TRAIN_MODE | NUM_LEVELS
    --------------|------------------------|------------------|------------|------------
    0             | 6033/1080/1111 (env)   | 0.json           | fallback   | 100000
    1-4           | same as ID             | {ID}.json        | random     | (not set)

    Default seeds by environment: coinrun=6033, maze=1080, heist=1111

Examples:
    $(basename "$0") -e coinrun -x 0
    $(basename "$0") --env maze --experiment 2
    $(basename "$0") --env heist --experiment 0
EOF
    exit 1
}

# Parse command line arguments
ENV_TYPE=""
EXPERIMENT_ID=""
RANDOMIZE_AGENT_START=false
RANDOM_PERCENT_OVERRIDE=""
NUM_TIMESTEPS=200000000
TRAIN_DAYS=3

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
        --randomize-agent-start)
            RANDOMIZE_AGENT_START=true
            shift
            ;;
        --random-percent)
            RANDOM_PERCENT_OVERRIDE="$2"
            shift 2
            ;;
        --num-timesteps)
            NUM_TIMESTEPS="$2"
            shift 2
            ;;
        --days)
            TRAIN_DAYS="$2"
            shift 2
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
if [ "$ENV_TYPE" != "coinrun" ] && [ "$ENV_TYPE" != "maze" ] && [ "$ENV_TYPE" != "heist" ]; then
    echo "Error: ENV_TYPE must be 'coinrun', 'maze', or 'heist', got '$ENV_TYPE'"
    exit 1
fi

if [ "$RANDOMIZE_AGENT_START" = true ] && [ "$ENV_TYPE" != "maze" ]; then
    echo "Error: --randomize-agent-start is currently supported only for maze"
    exit 1
fi

if [ -n "$RANDOM_PERCENT_OVERRIDE" ] && ! [[ "$RANDOM_PERCENT_OVERRIDE" =~ ^[0-9]+$ ]]; then
    echo "Error: --random-percent must be an integer, got '$RANDOM_PERCENT_OVERRIDE'"
    exit 1
fi

if [ -n "$RANDOM_PERCENT_OVERRIDE" ] && [ "$RANDOM_PERCENT_OVERRIDE" -gt 100 ]; then
    echo "Error: --random-percent must be between 0 and 100, got '$RANDOM_PERCENT_OVERRIDE'"
    exit 1
fi

if ! [[ "$NUM_TIMESTEPS" =~ ^[0-9]+$ ]]; then
    echo "Error: --num-timesteps must be a positive integer, got '$NUM_TIMESTEPS'"
    exit 1
fi

if [ "$NUM_TIMESTEPS" -le 0 ]; then
    echo "Error: --num-timesteps must be positive, got '$NUM_TIMESTEPS'"
    exit 1
fi

if ! [[ "$TRAIN_DAYS" =~ ^[0-9]+$ ]]; then
    echo "Error: --days must be a positive integer, got '$TRAIN_DAYS'"
    exit 1
fi

if [ "$TRAIN_DAYS" -le 0 ]; then
    echo "Error: --days must be positive, got '$TRAIN_DAYS'"
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
elif [ "$ENV_TYPE" = "heist" ]; then
    ENV_NAME="heist_afh"
    VAL_ENV_NAME="heist_afh"
    DEFAULT_SEED=1111
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
echo "  RANDOM_START:  $RANDOMIZE_AGENT_START"
echo "  TIMESTEPS:     $NUM_TIMESTEPS"
echo "  TRAIN_DAYS:    $TRAIN_DAYS"
echo ""

if [ -n "$RANDOM_PERCENT_OVERRIDE" ]; then
    RANDOM_PERCENTS=("$RANDOM_PERCENT_OVERRIDE")
fi

for random_percent in "${RANDOM_PERCENTS[@]}"; do
    exp_name="${EXP_PREFIX}_${ENV_TYPE}_exp${EXPERIMENT_ID}_${random_percent}p"
    if [ "$RANDOMIZE_AGENT_START" = true ]; then
        exp_name="${exp_name}_random_start"
    fi

    # Build optional arguments
    EXTRA_ARGS=""
    if [ -n "$NUM_LEVELS" ]; then
        EXTRA_ARGS="--num_levels $NUM_LEVELS"
    fi
    if [ "$RANDOMIZE_AGENT_START" = true ]; then
        EXTRA_ARGS="$EXTRA_ARGS --randomize_agent_start --validate_random_agent_start"
    fi

    echo "Submitting job: $exp_name"

    sbatch --qos=default \
        --gres=gpu:1 \
        --time=${TRAIN_DAYS}-00:00:00 \
        --mem=128G \
        --job-name="$exp_name" \
        --output="${LOG_DIR}/${exp_name}_%j.out" \
        --wrap="source ${CONDA_BASE}/etc/profile.d/conda.sh && cd $TRAIN_DIR && conda run -n $CONDA_ENV python train.py \
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
            --num_timesteps $NUM_TIMESTEPS \
            --log_interval 4000000 \
            --num_checkpoints 10 \
            --num_threads 4 \
            --seed $SEED \
            --logdir_base $CHECKPOINT_BASE \
            $EXTRA_ARGS"
done

echo ""
echo "All jobs submitted. Check logs in: $LOG_DIR"

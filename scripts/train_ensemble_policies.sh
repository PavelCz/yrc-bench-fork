#!/bin/bash

# Get script directory and project paths (resolve symlinks for slurm compatibility)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$(dirname "$SCRIPT_DIR")")"
TRAIN_DIR="${PROJECT_ROOT}/lib/train-procgen-pytorch"

# Default configuration
CONDA_ENV="ood-stable"
EXP_PREFIX="icml2_ensemble"
LEVEL_SEEDS_FOLDER="/nas/ucb/czempin/data/goal-misgen/seeds/icml"
LOG_DIR="/nas/ucb/czempin/data/goal-misgen/logs/train_ensemble_policies"
NUM_ENSEMBLE_MEMBERS=4

# Usage function
usage() {
    cat << EOF
Usage: $(basename "$0") -e ENV_TYPE -x EXPERIMENT_ID [OPTIONS]

Train ensemble policies for uncertainty-based OOD detection experiments.

Required arguments:
    -e, --env ENV_TYPE        Environment type: "coinrun", "maze", or "heist"
    -x, --experiment ID       Experiment ID: 0-4

Optional arguments:
    -m, --member ID           Train only a specific ensemble member (0-3)
    -h, --help                Show this help message

Experiment configurations:
    Each experiment trains 4 ensemble members, each with different training seeds.
    Seed files: ensemble-{EXPERIMENT_ID}_{MEMBER_ID}.json

    All ensemble members are trained with random_percent=0 (in-distribution only).

Examples:
    # Train all 4 ensemble members for experiment 0
    $(basename "$0") -e coinrun -x 0

    # Train only ensemble member 2 for experiment 1
    $(basename "$0") -e maze -x 1 -m 2
EOF
    exit 1
}

# Parse command line arguments
ENV_TYPE=""
EXPERIMENT_ID=""
MEMBER_ID=""

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
        -m|--member)
            MEMBER_ID="$2"
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
if [ "$ENV_TYPE" != "coinrun" ] && [ "$ENV_TYPE" != "maze" ] && [ "$ENV_TYPE" != "heist" ]; then
    echo "Error: ENV_TYPE must be 'coinrun', 'maze', or 'heist', got '$ENV_TYPE'"
    exit 1
fi

# Validate EXPERIMENT_ID is a number 0-2
if ! [[ "$EXPERIMENT_ID" =~ ^[0-4]$ ]]; then
    echo "Error: EXPERIMENT_ID must be an integer between 0 and 4, got '$EXPERIMENT_ID'"
    exit 1
fi

# Validate MEMBER_ID if provided
if [ -n "$MEMBER_ID" ]; then
    if ! [[ "$MEMBER_ID" =~ ^[0-3]$ ]]; then
        echo "Error: MEMBER_ID must be an integer between 0 and 3, got '$MEMBER_ID'"
        exit 1
    fi
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set environment-specific parameters
if [ "$ENV_TYPE" = "coinrun" ]; then
    ENV_NAME="coinrun"
    VAL_ENV_NAME="coinrun"
elif [ "$ENV_TYPE" = "maze" ]; then
    ENV_NAME="maze_afh"
    VAL_ENV_NAME="maze_afh"
elif [ "$ENV_TYPE" = "heist" ]; then
    ENV_NAME="heist_afh"
    VAL_ENV_NAME="heist_afh"
fi

echo "Starting ensemble training with:"
echo "  ENV_TYPE:      $ENV_TYPE"
echo "  EXPERIMENT_ID: $EXPERIMENT_ID"
if [ -n "$MEMBER_ID" ]; then
    echo "  MEMBER_ID:     $MEMBER_ID (single member)"
else
    echo "  MEMBERS:       0-$((NUM_ENSEMBLE_MEMBERS-1)) (all)"
fi
echo ""

# Determine which members to train
if [ -n "$MEMBER_ID" ]; then
    MEMBERS=("$MEMBER_ID")
else
    MEMBERS=($(seq 0 $((NUM_ENSEMBLE_MEMBERS-1))))
fi

for member in "${MEMBERS[@]}"; do
    # Seed file path: ensemble-{exp_id}_{member_id}.json
    LEVEL_SEEDS_FILE="ensemble-${EXPERIMENT_ID}_${member}.json"
    LEVEL_SEEDS_PATH="${LEVEL_SEEDS_FOLDER}/${LEVEL_SEEDS_FILE}"

    # Check if seed file exists
    if [ ! -f "$LEVEL_SEEDS_PATH" ]; then
        echo "Warning: Seed file not found: $LEVEL_SEEDS_PATH"
        echo "  Skipping ensemble member $member"
        continue
    fi

    exp_name="${EXP_PREFIX}_${ENV_TYPE}_exp${EXPERIMENT_ID}_m${member}"

    # Use member ID as seed for reproducibility (different seed per member)
    SEED=$((EXPERIMENT_ID * 10 + member + 5))

    echo "Submitting job: $exp_name"
    echo "  Seed file: $LEVEL_SEEDS_FILE"
    echo "  Seed: $SEED"

    sbatch --qos=default \
        --gres=gpu:1 \
        --time=3-00:00:00 \
        --mem=100G \
        --job-name="$exp_name" \
        --output="${LOG_DIR}/${exp_name}_%j.out" \
        --wrap="cd $TRAIN_DIR && conda run -n $CONDA_ENV python -m apps.train \
            --level_seeds_file ${LEVEL_SEEDS_PATH} \
            --train_mode random \
            --eval_mode sequential \
            --exp_name $exp_name \
            --env_name $ENV_NAME \
            --val_env_name $VAL_ENV_NAME \
            --random_percent 0 \
            --random_percent_val 50 \
            --distribution_mode hard \
            --param_name paper \
            --num_timesteps 200000000 \
            --log_interval 4000000 \
            --num_checkpoints 10 \
            --num_threads 4 \
            --seed $SEED"
done

echo ""
echo "All jobs submitted. Check logs in: $LOG_DIR"

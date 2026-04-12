#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --time=3:00:00
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem=100G
#SBATCH --cpus-per-task 30
#SBATCH --partition gpu

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ood

CALIBRATION_PATH=/tmp/test_ensemble_calibration.npz
CHECKPOINT_PATH=/tmp/test_ensemble_bin_0.npz

python calibrate_afhp.py -c configs/eval/coinrun/ensemble_variance.yaml \
    -en coinrun \
    -n test_ensemble \
    -sim /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/model_200015872.pth \
    -weak /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/model_200015872.pth \
    -strong /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_50p/2026-01-13__14-46-26__seed_6033/model_200015872.pth \
    --calibration_path ${CALIBRATION_PATH} \
    --experiment_group test_ensemble \
    -cp_ensemble_members \
        /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp1_0p/2026-01-15__13-30-21__seed_1/model_200015872.pth \
        /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp2_0p/2026-01-15__13-30-24__seed_2/model_200015872.pth

python eval_afhp_bin.py -c configs/eval/coinrun/ensemble_variance.yaml \
    -en coinrun \
    -n test_ensemble \
    -sim /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/model_200015872.pth \
    -weak /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/model_200015872.pth \
    -strong /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_50p/2026-01-13__14-46-26__seed_6033/model_200015872.pth \
    -num_bins 10 \
    --bin_idx 0 \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --calibration_path ${CALIBRATION_PATH} \
    --experiment_group test_ensemble \
    -cp_ensemble_members \
        /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp1_0p/2026-01-15__13-30-21__seed_1/model_200015872.pth \
        /project2/biyik_1165/czempin/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp2_0p/2026-01-15__13-30-24__seed_2/model_200015872.pth

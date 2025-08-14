#!/usr/bin/env bash

# If you change these, also change .circle/config.yml.
# Separate like this: SRC_FILES=(src/ tests/ setup.py)
SRC_FILES=(
    "YRC/algorithms/lightning_ae.py" 
    "YRC/policies/lightning_ae.py" 
    "YRC/core/dataset.py"
    "YRC/core/rollout_helper.py"
    "eval_thresholds.py"
    "YRC/policies/mahalanobis_ae.py"
    "analyzing/coinrun_counterfactual_analysis.py"
)
EXCLUDED_FILES=("")

# set -x  # echo commands
set -e  # quit immediately on error

# Run ruff as formatter (black-ish and isort-ish).
ruff format "${SRC_FILES[@]}" --exclude "${EXCLUDED_FILES[@]}"
# Run ruff as linter (flake8-ish).
ruff check "${SRC_FILES[@]}" --exclude "${EXCLUDED_FILES[@]}"
# Run pytype with suppressed debug logging
pytype "${SRC_FILES[@]}" --exclude "${EXCLUDED_FILES[@]}" --verbosity=0 2>/dev/null

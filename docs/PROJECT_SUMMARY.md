# YRC-Bench Fork: OOD Detection for Weak-to-Strong Coordination

## Overview

This repository is a fork of [YRC-Bench](https://github.com/modanesh/YRC-Bench) (Yield and Request Control Benchmark) focused on **Procgen environments** and **out-of-distribution (OOD) detection** methods for coordinating between weak and strong RL policies.

### The Core Problem

In reinforcement learning, we often have access to both:
- A **weak agent**: A policy trained on limited data or with fewer resources (e.g., trained on 0% goal randomization)
- A **strong agent**: An expert policy with better performance (e.g., trained on 50% goal randomization, or an oracle)

The coordination problem is: **When should the weak agent ask the strong agent for help?**

The goal is to maximize performance while minimizing the number of times we query the strong agent (which may be expensive, slow, or limited). This creates a trade-off measured by:
- **AFHP (Ask-For-Help Percentage)**: How often the weak agent defers to the strong agent
- **Performance**: The return/reward achieved

### Research Focus: OOD Detection for Coordination

This fork investigates using **out-of-distribution detection** as a coordination signal. The hypothesis: if the weak agent can detect when it's in an unfamiliar situation (OOD), it should ask for help.

The key insight is that weak agents trained on in-distribution (ID) data will encounter OOD situations at test time. Detecting these OOD states can trigger help requests.

## Experimental Setup

### Environments

The experiments use **Procgen** environments with controllable goal randomization:

| Environment | ID Condition | OOD Condition |
|-------------|--------------|---------------|
| `coinrun` | Coin at fixed location | Coin at random location |
| `maze_afh` | Cheese in corner | Cheese at random location |

The `random_percent` parameter controls the probability of goal randomization:
- `random_percent=0`: Always ID (deterministic goal placement)
- `random_percent=50`: 50/50 mix of ID and OOD
- `random_percent=100`: Always OOD (random goal placement)

### Policy Training

Policies are trained using [train-procgen-pytorch](https://github.com/jbkjr/train-procgen-pytorch) with PPO:

- **Weak policy**: Trained on `random_percent=0` (ID only)
- **Strong policy**: Trained on `random_percent=50` (mixed ID/OOD)
- **Ensemble members**: Multiple weak policies with different seeds (for ensemble methods)

### Coordination Methods

| Method | Description | Config Flag |
|--------|-------------|-------------|
| `max_prob` | Threshold on max softmax probability | `-cp_metric max_prob` |
| `max_logit` | Threshold on max logit value | `-cp_metric max_logit` |
| `ensemble_variance` | Variance of softmax outputs across ensemble | `-cp_metric ensemble_variance` |
| `svdd_image` | Deep SVDD on raw observations | `-cp_method DeepSVDD -cp_feature obs` |
| `svdd_latent` | Deep SVDD on hidden features | `-cp_method DeepSVDD -cp_feature hidden` |
| `ts_random` | Random per-timestep baseline | `timestep_random` config |
| `lb_random` | Random per-level baseline | `level_based_random` config |

## Repository Structure

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_eval.py` | Submit SLURM jobs for AFHP evaluation |
| `run_gather_rollouts.py` | Collect rollouts for OOD detector training |
| `run_train.py` | Train Deep SVDD coordination policies |
| `train_policies.sh` | Train weak/strong acting policies |
| `train_ensemble_policies.sh` | Train ensemble member policies |
| `sync_evals.sh` | Sync evaluation results between servers |

### Key Entry Points

| File | Purpose |
|------|---------|
| `eval_afhp.py` | Main evaluation script with adaptive threshold sampling |
| `gather_rollouts.py` | Collect rollouts from weak policy for OOD training |
| `train.py` | Train coordination policies (SVDD, RL-based) |

### Configuration (`configs/`)

```
configs/
├── common.yaml              # Shared parameters
├── procgen_*.yaml           # Procgen-specific configs
└── eval/
    ├── coinrun/
    │   ├── max_prob.yaml
    │   ├── max_logit.yaml
    │   ├── ensemble_variance.yaml
    │   └── ...
    └── maze/
        └── ...
```

### Analysis (`analyzing/`)

| Script | Purpose |
|--------|---------|
| `icml_plot.py` | Plot results with aggregation across experiments |
| `plot_ood.py` | Plot individual experiment results |
| `plot_histogram.py` | Visualize score distributions |

## Workflow

### 1. Train Acting Policies

```bash
# Train weak policy (0% randomization)
./scripts/train_policies.sh coinrun 0

# Train strong policy (50% randomization)
./scripts/train_policies.sh coinrun 50

# Train ensemble members (for ensemble_variance method)
./scripts/train_ensemble_policies.sh coinrun
```

### 2. Gather Rollouts (for SVDD methods)

```bash
python scripts/run_gather_rollouts.py \
    --server chai \
    --env coinrun \
    --prefix icml04 \
    --exp-ids 0 1 2
```

### 3. Train SVDD Coordination Policy

```bash
python scripts/run_train.py \
    --server chai \
    --env coinrun \
    --feature-type hidden \
    --prefix icml04 \
    --exp-ids 0 1 2
```

### 4. Evaluate Methods

```bash
# Evaluate max_prob method
python scripts/run_eval.py \
    --server chai \
    --env coinrun \
    --method max-prob \
    --prefix icml04 \
    --exp-ids 0 1 2

# Evaluate ensemble method
python scripts/run_eval.py \
    --server chai \
    --env coinrun \
    --method ensemble \
    --prefix icml04 \
    --num-ensemble-members 4
```

### 5. Plot Results

```bash
# List available methods
python -m analyzing.icml_plot \
    --eval_dir /path/to/evals \
    --prefix icml04 \
    --env coinrun \
    --list

# Generate aggregated plot
python -m analyzing.icml_plot \
    --eval_dir /path/to/evals \
    --prefix icml04 \
    --env coinrun \
    --x_data_key afhp \
    --y_data_key performance \
    --save results.png
```

## Key Metrics

### Evaluation Output

Each evaluation produces an `.npz` file containing:
- `afhps`: Ask-for-help percentages at different thresholds
- `performances`: Corresponding performance values
- `meta`: Detailed per-episode information including:
  - `level_ood_gt`: Ground truth OOD labels
  - `level_ood_pred`: Predicted OOD labels (based on threshold)
  - `raw_returns`: Episode returns
  - `episode_lengths`: Episode durations

### Derived Metrics

| Metric | Description |
|--------|-------------|
| `ood_pred_percentage` | Fraction of episodes predicted as OOD |
| `ood_accuracy` | Accuracy of OOD predictions vs ground truth |
| `true_positive` | TPR for OOD detection |
| `false_positive` | FPR for OOD detection |

## Server Configuration

The scripts support multiple server configurations:

| Server | Checkpoint Base | Seeds Base |
|--------|-----------------|------------|
| `chai` | `/nas/ucb/czempin/data/goal-misgen/policy/icml` | `/nas/ucb/czempin/data/goal-misgen/seeds/icml` |
| `snoopy` | `/scr/pavel/data/goal-misgen/policy/icml` | `/scr/pavel/data/goal-misgen/seeds/icml` |

## Experiment Naming Convention

```
{prefix}_{env}_exp{id}
```

Examples:
- `icml04_coinrun_exp0`: ICML experiment 4, coinrun environment, experiment ID 0
- `icml04_maze_exp1`: ICML experiment 4, maze environment, experiment ID 1

Each experiment ID corresponds to a different random seed for policy training.

## References

- Original YRC-Bench paper: [Learning to Coordinate with Experts](https://arxiv.org/abs/2502.09583)
- Goal Misgeneralization in Procgen: [Original Paper](https://arxiv.org/abs/2105.14111)
- Deep SVDD: [Deep One-Class Classification](http://proceedings.mlr.press/v80/ruff18a.html)

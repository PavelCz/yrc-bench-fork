# Agent Instructions

This file is shared via symlink as `CLAUDE.md` and `Agents.md`.
It provides guidance to coding agents working in this repository.

## Project Overview

This repository is a fork of YRC-Bench used for the "Getting by Goal Misgeneralization with a Little Help From an Expert" experiments.
The current fork is centered on **Procgen**, especially the paper workflows for `coinrun` and `maze`.

MiniGrid and Cliport code from upstream still exists, but most active work in this fork is around:
- `YRC/envs/procgen/`
- `configs/procgen_*.yaml`
- `configs/eval/coinrun/` and `configs/eval/maze/`
- `scripts/run_eval.py`, `scripts/train_svdd.py`, and `scripts/common.py`
- `docs/percentile_calibration.md` and `docs/bin_threshold_search.md`

## Reflective Memories

- When writing conclusions or documentation, avoid language that assigns subjective value to an algorithm or design unless the repo contains evidence for that claim.

## Development Environment

There are two conda environments:
- **`ood`**: development, debugging, linting, and local tests
- **`ood-stable`**: paper-style runs and SLURM job scripts

```bash
# Development
conda run -n ood <command>

# Or activate the environment first
conda activate ood

# Paper / batch jobs
conda run -n ood-stable <command>
```

## CLI Conventions

- Use `tyro` for standalone script CLIs under `scripts/` and `analyzing/`.
- Keep config-driven app entrypoints on `flags.py` / `jsonargparse`.
- See `docs/cli_conventions.md` for the repo convention and migration rule.

## Key Commands

### Code Quality

Prefer direct per-file checks on the files you touched:

```bash
conda run -n ood ruff format <files>
conda run -n ood ruff check <files>
conda run -n ood pytype <files>
```

`ci/format_and_check.sh` exists, but its hard-coded file list includes some legacy `analyzing/` paths. Inspect it before relying on it as the canonical check command.

### Training

`train.py` is the root training entrypoint for model-based coordination methods that train from rollouts without threshold search. In the current code, that means `general.algorithm` values in:
- `ood`
- `lightning_ae`

Typical invocation:

```bash
python train.py -c configs/procgen_ood.yaml -n RUN_NAME -en ENV_NAME \
    -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
    -query_cost COST -cp_feature obs
```

Paper-specific automation lives in:

```bash
python scripts/train_svdd.py --env coinrun --method svdd-latent --exp-ids 0 1 2 3
./scripts/train_policies.sh --env coinrun --experiment 0
./scripts/train_ensemble_policies.sh --env coinrun --experiment 0
```

### Evaluation

There are multiple evaluation entrypoints with different purposes:

```bash
# Evaluate an underlying acting policy checkpoint directly
python eval_policy.py -c configs/procgen_threshold.yaml \
    --model_file YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth

# Calibrate percentile-to-threshold mapping for AFHP evaluation
python calibrate_afhp.py --coordination_artifact_dir PATH/TO/ARTIFACT_DIR [...]

# Evaluate one AFHP bin after calibration
python eval_afhp_bin.py --bin_idx 0 \
    --checkpoint_path PATH/TO/bin_0.npz \
    --calibration_path PATH/TO/calibration.npz [...]
```

`scripts/run_eval.py` is the SLURM wrapper that assembles calibration and AFHP bin jobs for the paper workflows.

### Analysis

The active analysis scripts for this fork are module-style entrypoints under `analyzing/`:

```bash
python -m analyzing.icml_plot ...
python -m analyzing.plot_ood_rate ...
python -m analyzing.episode_length_dist ...
python -m analyzing.plot_policy_training_curves ...
```

The original YRC-Bench aggregation scripts are kept under `analyzing/yrc_bench/`:

```bash
python analyzing/yrc_bench/parse.py
python analyzing/yrc_bench/aggregate.py
python analyzing/yrc_bench/fig*.py
```

### Installation

```bash
pip install -r requirements.txt
pip install -r requirements_minigrid.txt  # only if you need MiniGrid
pip install -e lib/LIBRARY_NAME
```

## Architecture Overview

### Core Components

1. **`YRC/algorithms/`**: coordination algorithm implementations
   - `ood.py`: DeepSVDD-style OOD training
   - `lightning_ae.py`: Lightning autoencoder training
   - `rl.py`: PPO-based coordination
   - `threshold.py`: confidence-threshold methods
   - `random.py`: random and always baselines

2. **`YRC/core/`**: shared runtime and evaluation infrastructure
   - `algorithm.py`: base algorithm interface
   - `dataset.py`: rollout dataset handling
   - `environment.py`: environment factory
   - `policy.py`: policy factory and interfaces
   - `evaluator.py`: rollout/evaluation loop
   - `eval_setup.py`: shared runtime for AFHP evaluation entrypoints
   - `eval_calibration.py`: percentile calibration
   - `artifacts.py`: coordination artifact path helpers
   - `configs/`: config loading and global config helpers

3. **`YRC/coverage/`**: AFHP coverage search
   - `coverage_search.py`: bin-based threshold search with restartable checkpoints

4. **`YRC/envs/`**: benchmark-specific wrappers and acting policy loaders
   - `procgen/`: primary focus in this fork
   - `minigrid/` and `cliport/`: upstream code still present

5. **`YRC/policies/`**: coordination policies
   - `base.py`: `TimestepRandomPolicy`, `LevelBasedRandomPolicy`, `AlwaysPolicy`
   - `heuristic.py`: `ExponentialHeuristicPolicy`, `WaitPolicy`
   - `threshold.py`: `ThresholdPolicy`
   - `ood.py`: `OODPolicy`
   - `lightning_ae.py`: `LightningAEPolicy`
   - `mahalanobis_ae.py`: Mahalanobis detector support
   - `rl.py`: RL coordination policies

### Configuration System

The project uses hierarchical YAML configs in `configs/`:
- `common.yaml`: shared defaults
- `procgen_*.yaml`: top-level Procgen configs
- `configs/eval/{coinrun,maze}/`: paper evaluation configs by method
- command-line overrides from `flags.py`

### Important Implementation Details

1. **AFHP evaluation is bin-based**
   - `calibrate_afhp.py` saves calibration artifacts
   - `eval_afhp_bin.py` evaluates one equal-width AFHP bin at a time
   - `YRC/coverage/coverage_search.py` starts from the policy's calibrated percentile heuristic, then refines with binary search
   - per-bin `.npz` files make failed-bin restarts cheap

2. **Percentile calibration is central**
   - two AFHP metrics exist: `step_afhp` and `level_afhp`
   - policies implement `train_percentile_step()` and/or `train_percentile_level()`
   - see `docs/percentile_calibration.md` for the support matrix and formulas

3. **Feature type names in code**
   - `obs`
   - `hidden`
   - `dist`
   - `hidden_obs`
   - `hidden_dist`
   - `obs_dist`
   - `obs_hidden_dist`

4. **Threshold metrics in code**
   - `max_logit`
   - `max_prob`
   - `margin`
   - `neg_entropy`
   - `neg_energy`
   - `ensemble_variance`

5. **Paper automation is environment-limited**
   - `scripts/common.py` currently only enumerates `coinrun` and `maze`
   - method names such as `max-prob`, `max-logit`, `ts-random`, `svdd-image`, `svdd-latent`, `ensemble`, `ensemble-single`, and `wait` are defined there

6. **Checkpoint expectations**
   - acting policy checkpoints are loaded via `-sim`, `-weak`, and `-strong`
   - coordination model checkpoints are loaded from `config.experiment_dir / config.file_name`
   - AFHP calibration artifacts are typically stored as `calibration.npz` under a coordination artifact directory

## Python Best Practices

- Use `pathlib.Path` for filesystem work
- Follow `ruff` formatting and linting
- Add type hints where practical
- Prefer updating docs when changing evaluation workflow, calibration behavior, or script entrypoints

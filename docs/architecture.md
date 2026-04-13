# Architecture Overview

This repository has three main layers:

## `YRC/`

`YRC/` is the internal library code. It contains the reusable implementation for:

- coordination algorithms in `YRC/algorithms/`
- shared runtime and config logic in `YRC/core/`
- coordination policies in `YRC/policies/`
- benchmark adapters in `YRC/envs/`
- AFHP coverage search in `YRC/coverage/`

Most code changes that affect training or evaluation behavior should land here.

## `apps/`

`apps/` contains runnable Python entrypoints built on top of `flags.py` and the
`YRC/` package. These are the scripts you run directly when you want to execute
training, evaluation, calibration, or seed generation locally.

Current entrypoints include:

- `python -m apps.train`
- `python -m apps.gather_rollouts`
- `python -m apps.eval_policy`
- `python -m apps.calibrate_afhp`
- `python -m apps.eval_afhp_bin`
- `python -m apps.eval_strong_on_help`
- `python -m apps.generate_level_seeds`
- `python -m apps.generate_ensemble_seeds`

## `scripts/`

`scripts/` contains convenience wrappers and SLURM orchestration for paper and
cluster workflows. These scripts typically do not implement the core algorithmic
behavior themselves. Instead, they resolve paths, assemble commands, and submit
jobs that call the entrypoints under `apps/`.

Important examples:

- `scripts/run_eval.py` builds the calibration + AFHP bin workflow
- `scripts/train_svdd.py` submits rollout-gathering and SVDD training jobs
- `scripts/prep.py` contains shared SLURM command builders

## Supporting Directories

- `configs/` contains YAML experiment and infrastructure configuration
- `docs/` contains workflow and architecture notes
- `analyzing/` contains analysis and plotting entrypoints
- `tests/` contains targeted regression tests for workflow code

## Design Intent

The current layout separates concerns as follows:

- put reusable implementation in `YRC/`
- put direct user-facing app entrypoints in `apps/`
- put job submission and workflow automation in `scripts/`

This keeps cluster automation out of the core package while avoiding a crowded
repo root full of runnable scripts.

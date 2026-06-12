# Running smoke tests on cluster1

The `cluster1` server is the primary place to run end-to-end smoke tests
against the frozen experiment conda env. Use it when you need to validate
a sampler / eval / policy change before launching a larger sweep, or to
run one-off diagnostic scripts against real checkpoints.

## Prerequisites

- SSH access to `cluster1` configured as host alias `cluster1`.
- The branch under test must be pushed to `origin/ood`. Smoke jobs pull
  from `origin/ood` on cluster1 before submitting.
- Conda env `ood-stable` must exist at
  `/path/to/cluster1/anaconda3/envs/ood-stable` (frozen experiment env).
- The repo on cluster1 is at `/path/to/cluster1/code/goal-misgen/yrc-bench-fork`.

## Pulling the latest commit

```bash
ssh cluster1 'cd /path/to/cluster1/code/goal-misgen/yrc-bench-fork \
  && git fetch origin ood \
  && git pull --ff-only \
  && git log --oneline -3'
```

Confirm the expected commit SHA is at the top before submitting.

## Submitting an eval smoke job

The standard pattern is to invoke `scripts/run_eval.py` over SSH inside
`ood-stable`. The flags below are the smoke-test defaults — small enough
to finish quickly while still exercising the calibration + sampler path
end to end:

```bash
ssh cluster1 'cd /path/to/cluster1/code/goal-misgen/yrc-bench-fork \
  && source /path/to/cluster1/anaconda3/etc/profile.d/conda.sh \
  && conda activate ood-stable \
  && python scripts/run_eval.py \
       --env <coinrun|maze|...> \
       --method <method-key> \
       --prefix <experiment-prefix> \
       --exp-ids 0 \
       --num-levels 16 \
       --coverage-fraction 0.2 \
       --video-episodes 0 \
       --video-filter all \
       --cp-rolling-average none \
       --video-logging-mode folder \
       --video-filter-mode any'
```

Notes on the defaults:

- `--num-levels 16` — number of OOD-eval levels each threshold is scored
  against (not the total sampler iteration count).
- `--coverage-fraction 0.2` — coarse bin granularity for the AFHP
  coverage sampler, so fewer iterations are needed to mark coverage as
  complete.
- `--video-episodes 0` — skip video recording.
- `--calibration-levels` is omitted, so calibration uses the full
  validation split. Pass `--calibration-levels 64` to override
  (e.g. for image-SVDD smoke tests).
- `--exp-ids 0` runs one experiment; extend to a list for more coverage.

The script prints `Submitted <job_name>: Submitted batch job <JOBID>`.
Record `JOBID`.

Add `--dry-run` to preview the sbatch script without submitting.

## Log locations

For a job submitted with `--prefix PFX`, the two SLURM files land at

```
/path/to/cluster1/data/goal-misgen/slurm-logs/default/<PFX>/<YYYY-MM-DD>/<job_name>_<JOBID>.err
/path/to/cluster1/data/goal-misgen/slurm-logs/default/<PFX>/<YYYY-MM-DD>/<job_name>_<JOBID>.out
```

The `.err` file holds the per-run log (everything from `logging.info`,
including the eval tracker's per-iteration lines and sampler
diagnostics). The `.out` file holds only the launcher shell's stdout,
which is mostly empty for a still-running job because Python stdout is
block-buffered.

`<job_name>` follows `run_eval.py`'s convention,
`{env}_{method}[_{robust_key}]_exp{exp_id}` (for example
`coinrun_svdd-image_exp0` or `maze_max-prob_robust200_exp3`).

## Tailing in real time

```bash
ssh cluster1 'tail -F /path/to/cluster1/data/goal-misgen/slurm-logs/default/<PFX>/<YYYY-MM-DD>/<job_name>_<JOBID>.err'
```

## Cancelling

```bash
ssh cluster1 'scancel <JOBID>'
```

## Typical timing on cluster1 (`ood-stable` env, single GPU)

- Calibration: about 2 minutes for 64 episodes.
- Per evaluation (any threshold): about 1 to 2 minutes.

Scale expectations with `--num-levels` and `--calibration-levels`.

## Running an arbitrary script on cluster1

When you need to run a one-off diagnostic rather than a full eval batch,
reuse the same SSH/conda wrapper and call the script directly:

```bash
ssh cluster1 'cd /path/to/cluster1/code/goal-misgen/yrc-bench-fork \
  && source /path/to/cluster1/anaconda3/etc/profile.d/conda.sh \
  && conda activate ood-stable \
  && python <path/to/script.py> [args]'
```

A GPU is not strictly required for inference-only diagnostics but is
much faster.

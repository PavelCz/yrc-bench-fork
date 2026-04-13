# Adaptive Coverage Sampling

This document describes how the evaluation pipeline samples thresholds to build AFHP-vs-return curves.

## Overview

The goal is to efficiently sample enough thresholds to produce a well-covered curve of **Ask-For-Help Percentage (AFHP)** vs **return**. Rather than evaluating on a fixed grid, the system uses a binary search strategy to adaptively place evaluations where gaps exist.

## Code Path

```
scripts/run_eval.py
  → python -m apps.calibrate_afhp
  → python -m apps.eval_afhp_bin
    → YRC/coverage/coverage_search.py  (creates sampler + callbacks)
      → lib/acs/src/acs/sampler.py     (BinarySearchSampler)
        → YRC/core/evaluator.py        (runs episodes per threshold)
```

## Parameters

| Parameter | Default | Set In | Meaning |
|---|---|---|---|
| `coverage_fraction` | 0.05 | `scripts/run_eval.py` | Max normalized neighbor gap allowed on output axis |
| `max_total_evals` | 200 | bin-eval worker path | Hard budget of evaluations |
| `num_levels` | 5000 | `scripts/run_eval.py` | Episodes per evaluation |
| `threshold_sampler` | `"step_afhp"` | Config YAML | Which output axis to cover (`"step_afhp"` or `"level_afhp"`) |

`coverage_fraction` controls granularity: `num_bins = int(1.0 / coverage_fraction)`. With the default of 0.05, this yields 20 bins.

## Algorithm: BinarySearchSampler

The sampler divides the output axis (0–100% AFHP) into `num_bins` equal-width bins, then fills them:

### Phase 1: Evaluate extremes

- **Lower extreme**: threshold = +inf (never ask for help, AFHP ~ 0%)
- **Upper extreme**: threshold = -inf (always ask for help, AFHP ~ 100%)

Each extreme is placed into its corresponding output bin.

### Phase 2: Binary search fill

Recursively:
1. Pick the midpoint percentile between two already-evaluated percentiles.
2. Convert the percentile to a threshold via `policy.train_percentile_level()` or `policy.train_percentile_step()`.
3. Run a full evaluation (all `num_levels` episodes) at that threshold.
4. Place the result into its output bin based on the observed AFHP.
5. Recurse into left and right sub-intervals where empty bins remain.

### Stopping conditions

The algorithm stops when:
- All `num_bins` bins contain at least one sample, **or**
- The hard budget of `max_total_evals` is exhausted.

In practice, with 20 bins, the algorithm typically runs 20–40 evaluations.

## Evaluation Callbacks

The sampler doesn't know about environments or policies directly. Instead, `YRC/coverage/coverage_search.py` provides three callbacks:

- **`eval_at_percentile(p)`**: Converts percentile `p` (0–1) to a threshold via the policy's training distribution, runs evaluation, returns `(afhp%, mean_return, metadata)`.
- **`eval_at_lower_extreme()`**: Evaluates with threshold = +inf (never ask for help).
- **`eval_at_upper_extreme()`**: Evaluates with threshold = -inf (always ask for help).

## Per-Evaluation Metrics

Each evaluation runs `num_levels` episodes and computes (in `YRC/core/evaluator.py`):

- `action_1_frac` — fraction of steps where help was requested (AFHP axis)
- `env_return_mean` — average episodic return (performance axis)
- `level_seeds` — which level seeds were used
- `level_ood_pred` — per-episode OOD predictions (used downstream by `python -m apps.eval_strong_on_help`)

## Output

Results are saved to an NPZ file containing:
- All sampled curve points (threshold, AFHP, return)
- Per-point metadata including per-episode predictions
- Coverage statistics (max normalized gap on the output axis)

## Coverage Metric

The coverage metric is the **max normalized neighbor gap**: the largest gap between adjacent samples on the output axis, divided by the total output range. With `coverage_fraction=0.05`, the target is that no gap exceeds 5% of the output range.

## Percentile Calibration

For threshold-based methods (`max_prob`, `max_logit`, `ensemble_variance`), the sampler works in percentile space of the training score distribution. Before sampling begins, `python -m apps.calibrate_afhp` runs `policy.generate_scores()` on the calibration environment to collect per-step OOD scores and per-episode max scores. `train_percentile_step(p)` maps percentile `p` to a threshold via `np.percentile(step_scores, p)` (calibrated for step_afhp), while `train_percentile_level(p)` uses `np.percentile(episode_max_scores, p)` (calibrated for level_afhp).

For `TimestepRandomPolicy`, there is no OOD score distribution — the "score" is `torch.rand()`. However, the mapping from per-step help probability to per-episode OOD percentage is nonlinear: with probability `p` per step and episode length `L`, the fraction of episodes with any help request is `1 - (1-p)^L`. To account for this, `python -m apps.calibrate_afhp` runs a calibration step before sampling: it evaluates the weak agent alone (prob=0) on the calibration environment to measure the mean episode length, then `train_percentile_level` uses the inverse formula `prob = 1 - (percentile/100)^(1/L)`. `train_percentile_step` uses a simple linear mapping instead.

For `LevelBasedRandomPolicy`, the decision is per-episode, so `level_afhp` equals the help probability directly — no calibration is needed.

## Sampler Variants

The `threshold_sampler` config option selects the output axis:

- `"step_afhp"`: Covers the step AFHP axis (% of steps where help is requested).
- `"level_afhp"`: Covers the level AFHP axis (% of episodes where help is requested).

Both use the same `BinarySearchSampler` algorithm; they differ only in which metric defines the output bins.

There is also a `WaitPolicyAwareSampler` (in `lib/acs/src/acs/wait_policy_sampler.py`) that extends the binary search to detect regions where the output doesn't change despite varying the threshold — it declares these "unfillable" and stops searching there.

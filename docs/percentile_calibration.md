# Percentile Calibration

This document describes how policies map percentiles to thresholds for AFHP evaluation.

## Two AFHP Metrics

- **step_afhp**: fraction of *timesteps* where help was requested
- **level_afhp**: fraction of *episodes* where help was requested at least once

These require different calibrations. A single high-scoring timestep makes the entire episode count for level_afhp, so targeting 10% level_afhp requires far fewer per-step help requests than targeting 10% step_afhp.

## API

All policies implement two methods:

- `train_percentile_step(percentile)` — returns a threshold calibrated for step_afhp
- `train_percentile_level(percentile)` — returns a threshold calibrated for level_afhp

The `percentile` argument uses numpy convention: `percentile=90` means "90th percentile of scores", which corresponds to a target AFHP of 10% (only 10% of samples exceed this threshold).

Not all policies support both. Unsupported variants raise `NotImplementedError`.

## Policy Support Matrix

| Policy | `train_percentile_step` | `train_percentile_level` |
|---|---|---|
| `ThresholdPolicy` | per-step score percentiles | per-episode max score percentiles |
| `TimestepRandomPolicy` | linear mapping | `1 - p^(1/L)` formula using mean episode length |
| `LevelBasedRandomPolicy` | `NotImplementedError` | linear mapping |
| `ExponentialHeuristicPolicy` | `NotImplementedError` | `1 - p^(2/(L(L-1)))` formula using mean episode length |
| `WaitPolicy` | timestep threshold from episode length | empirical episode length percentiles |
| `OODPolicy` | per-step scores (rollout or training) | per-episode max scores |
| `LightningAEPolicy` | per-step scores (rollout or training) | per-episode max scores |

## Calibration Data

Calibration happens at the start of `eval_afhp.py`, before the sampler runs. The data comes from the fixed-seed calibration environment split `cal`. In the standard Procgen pipeline, `eval_afhp.py` maps the seed-file `validation` split to `cal`; see [level_seed_splits.md](level_seed_splits.md).

There are two calibration mechanisms depending on the policy type:

### Score-based calibration (ThresholdPolicy, OODPolicy, LightningAEPolicy)

These policies have an explicit score distribution that needs to be collected.

**ThresholdPolicy**: `eval_afhp.py` calls `policy.generate_scores(cal_env, num_cal_episodes)`, which runs one episode per validation seed in the calibration environment using the weak agent. During each episode, the policy computes its OOD score (e.g., `max_prob`, `max_logit`, `ensemble_variance`) at every timestep. This produces:
- `_train_scores`: all per-step scores across all episodes (flat array)
- `_train_episode_max_scores`: the maximum score within each episode (one value per episode)

**OODPolicy / LightningAEPolicy**: These policies support two sources of scores. During model training (`train_svdd.py`), per-step decision scores are collected (`clf.decision_scores_` / `_train_decision_scores`), but these lack episode boundaries. To support `train_percentile_level`, `eval_afhp.py` calls `policy.generate_scores()` which runs rollouts in the calibration environment with the trained OOD detector, collecting both per-step scores and per-episode max scores - the same approach as ThresholdPolicy. When rollout-based scores are available, `train_percentile_step` uses them instead of the training-time scores.

### Episode-length calibration (TimestepRandomPolicy, ExponentialHeuristicPolicy)

These policies don't have OOD scores — their "threshold" is a probability parameter. To calibrate the nonlinear mapping between per-step probability and per-episode help rate, `eval_afhp.py` measures the mean episode length:

1. Set the policy to never ask for help (probability = 0), so only the weak agent acts
2. Run a full evaluation on the calibration split using `evaluator.eval(policy, cal_envs, ["cal"])`
3. Extract `episode_length_mean` from the evaluation summary
4. Store it as `policy._mean_episode_length`
5. Restore the original probability

This mean episode length `L` is then used in the closed-form formulas in `train_percentile_level`.

Note: using the mean episode length is an approximation. A more accurate approach would use the empirical distribution of episode lengths and numerically invert `mean_i[1 - (1-prob)^L_i]`, but the mean-based formula is sufficient for now.

### Episode-length distribution calibration (WaitPolicy)

WaitPolicy asks for help at every timestep `t >= n`, so an episode has help iff its length exceeds `n`. To calibrate `train_percentile_level`, `eval_afhp.py` runs the weak agent alone (threshold set very high so it never asks) on the calibration split, and stores the full array of per-episode lengths as `policy._episode_lengths`. Then `train_percentile_level(p)` returns `np.percentile(episode_lengths, p)` - the p-th percentile of episode lengths is exactly the threshold where (100-p)% of episodes are long enough to receive help.

`train_percentile_step` still uses `max_episode_length` from config (not from data).

### No calibration needed (LevelBasedRandomPolicy)

**LevelBasedRandomPolicy** decides once per episode, so level_afhp equals the probability directly — no calibration needed.

## How Each Policy Works

### ThresholdPolicy (`YRC/policies/threshold.py`)

`train_percentile_step(p)` returns `np.percentile(_train_scores, p)`.

`train_percentile_level(p)` returns `np.percentile(_train_episode_max_scores, p)`. This works because a threshold set at the p-th percentile of episode-max scores means exactly (100-p)% of episodes have a max score exceeding the threshold — which is the definition of level_afhp.

### TimestepRandomPolicy (`YRC/policies/base.py`)

The "threshold" is the per-step help probability itself.

`train_percentile_step(p)` uses a linear mapping: `prob = (100 - p) / 100`.

`train_percentile_level(p)` accounts for the nonlinear relationship between per-step probability and per-episode help rate. With per-step probability `prob` and episode length `L`, the probability that an episode has *at least one* help step is `1 - (1-prob)^L`. The inverse gives `prob = 1 - (p/100)^(1/L)`.

### LevelBasedRandomPolicy (`YRC/policies/base.py`)

Decides once per episode whether to ask for help, so level_afhp equals the probability directly.

`train_percentile_level(p)` uses a linear mapping: `prob = (100 - p) / 100`.

`train_percentile_step` is not supported — per-step calibration doesn't apply to a per-episode decision.

### ExponentialHeuristicPolicy (`YRC/policies/heuristic.py`)

At timestep `t`, the probability of asking for help is `1 - (1 - ood_starting_prob)^t`. The probability of no help in an entire episode of length `L` is:

```
P(no help) = product_{t=0}^{L-1} (1 - ood_starting_prob)^t = (1 - ood_starting_prob)^{L(L-1)/2}
```

`train_percentile_level(p)` inverts this: `ood_starting_prob = 1 - (p/100)^{2/(L(L-1))}`. This is analogous to `TimestepRandomPolicy`'s formula but with exponent `2/(L(L-1))` instead of `1/L`, because the per-step probability grows over time.

`train_percentile_step` is not supported.

### WaitPolicy (`YRC/policies/heuristic.py`)

Waits `n` timesteps, then always asks for help. The threshold is the number of timesteps to wait. An episode has help iff its length > threshold.

`train_percentile_step(p)` maps linearly: `threshold = max_episode_length * p / 100`.

`train_percentile_level(p)` returns `np.percentile(episode_lengths, p)` using the empirical episode length distribution from calibration data.

### OODPolicy and LightningAEPolicy

These policies now support both calibration methods via `generate_scores()`, which runs rollouts with the trained OOD detector to collect per-step and per-episode-max scores — the same approach as ThresholdPolicy.

`train_percentile_step(p)` uses rollout-based per-step scores if available, otherwise falls back to decision scores from model training.

`train_percentile_level(p)` returns `np.percentile(episode_max_scores, p)` from the rollout data.

`LightningAEPolicy` inherits `generate_scores()` and `_rollout_once()` from `OODPolicy`, overriding `_compute_scores()` to use its reconstruction-error scoring instead of `clf.decision_function()`.

## Where These Are Called

The samplers in `YRC/coverage/coverage_search.py` call the appropriate method based on the sampler type:

- `create_level_afhp_threshold_sampler` → calls `train_percentile_level`
- `create_step_afhp_threshold_sampler` → calls `train_percentile_step`

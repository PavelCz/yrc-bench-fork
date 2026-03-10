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
| `WaitPolicy` | timestep threshold from episode length | `NotImplementedError` |
| `OODPolicy` | training decision scores | `NotImplementedError` |
| `LightningAEPolicy` | training decision scores | `NotImplementedError` |

## How Each Policy Works

### ThresholdPolicy (`YRC/policies/threshold.py`)

`generate_scores()` runs rollouts with the weak agent and collects:
- `_train_scores`: all per-step OOD scores (flat list across all episodes)
- `_train_episode_max_scores`: the max score within each episode

`train_percentile_step(p)` returns `np.percentile(_train_scores, p)`.

`train_percentile_level(p)` returns `np.percentile(_train_episode_max_scores, p)`. This works because a threshold set at the p-th percentile of episode-max scores means exactly (100-p)% of episodes have a max score exceeding the threshold — which is the definition of level_afhp.

### TimestepRandomPolicy (`YRC/policies/base.py`)

The "threshold" is the per-step help probability itself.

`train_percentile_step(p)` uses a linear mapping: `prob = (100 - p) / 100`.

`train_percentile_level(p)` accounts for the nonlinear relationship between per-step probability and per-episode help rate. With per-step probability `prob` and episode length `L`, the probability that an episode has *at least one* help step is `1 - (1-prob)^L`. The inverse gives `prob = 1 - (p/100)^(1/L)`, where `L` is the mean episode length measured during a calibration step in `eval_afhp.py`.

Note: using the mean episode length is an approximation. A more accurate approach would use the empirical distribution of episode lengths and numerically invert `mean_i[1 - (1-prob)^L_i]`, but the mean-based formula is sufficient for now.

### LevelBasedRandomPolicy (`YRC/policies/base.py`)

Decides once per episode whether to ask for help, so level_afhp equals the probability directly.

`train_percentile_level(p)` uses a linear mapping: `prob = (100 - p) / 100`.

`train_percentile_step` is not supported — per-step calibration doesn't apply to a per-episode decision.

### ExponentialHeuristicPolicy (`YRC/policies/heuristic.py`)

At timestep `t`, the probability of asking for help is `1 - (1 - ood_starting_prob)^t`. The probability of no help in an entire episode of length `L` is:

```
P(no help) = product_{t=0}^{L-1} (1 - ood_starting_prob)^t = (1 - ood_starting_prob)^{L(L-1)/2}
```

`train_percentile_level(p)` inverts this: `ood_starting_prob = 1 - (p/100)^{2/(L(L-1))}`, where `L` is the mean episode length measured during calibration. This is analogous to `TimestepRandomPolicy`'s formula but with exponent `2/(L(L-1))` instead of `1/L`, because the per-step probability grows over time.

`train_percentile_step` is not supported.

### WaitPolicy (`YRC/policies/heuristic.py`)

Waits `n` timesteps, then always asks for help. The threshold is the number of timesteps to wait.

`train_percentile_step(p)` maps linearly: `threshold = max_episode_length * p / 100`.

`train_percentile_level` is not supported.

### OODPolicy and LightningAEPolicy

Use per-step decision scores from model training (`clf.decision_scores_` or `_train_decision_scores`).

`train_percentile_step(p)` returns `np.percentile(decision_scores, p)`.

`train_percentile_level` is not supported — fixing this would require tracking episode boundaries during model training.

## Where These Are Called

The samplers in `YRC/coverage/coverage_search.py` call the appropriate method based on the sampler type:

- `create_level_afhp_threshold_sampler` → calls `train_percentile_level`
- `create_step_afhp_threshold_sampler` → calls `train_percentile_step`

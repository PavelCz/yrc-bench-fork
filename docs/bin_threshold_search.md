# Bin Threshold Search

This document describes how each parallel worker finds the policy threshold that
lands in its assigned AFHP bin. The implementation is in `run_bin()` in
`YRC/coverage/coverage_search.py`.

## Setup

The AFHP space `[0, 1]` is divided into `num_bins` equal-width bins. Bin `i`
covers the range:

```
bin_lo = i / num_bins
bin_hi = (i + 1) / num_bins
target_afhp = (bin_lo + bin_hi) / 2   # midpoint
```

Each worker's job is to find a threshold such that the policy's observed AFHP
falls inside `[bin_lo, bin_hi]`.

## Key relationship: AFHP and percentile are inversely related

For `level_afhp`, a threshold T produces AFHP equal to the fraction of
episodes whose maximum per-step score exceeds T. So:

- **Higher threshold → fewer episodes exceed it → lower AFHP**
- **Lower threshold → more episodes exceed it → higher AFHP**

The calibration step (see `docs/percentile_calibration.md`) maps percentiles of
the training score distribution to thresholds. Percentile `p` gives the
threshold that the top `(100 - p)%` of training episodes exceed. So for a
target AFHP of `a`:

```
percentile = 100 - a * 100
threshold  = train_percentile_level(percentile)
```

This means AFHP and percentile move in **opposite directions**: a higher
percentile produces a higher threshold, which produces a lower AFHP.

A bin covering AFHP range `[bin_lo, bin_hi]` therefore corresponds to the
**percentile bracket**:

```
lo_pct = 100 - bin_hi * 100   # low percentile → high AFHP (upper edge)
hi_pct = 100 - bin_lo * 100   # high percentile → low AFHP (lower edge)
```

## Step 1: Initial heuristic guess

The worker converts the target AFHP directly to a percentile and queries the
policy's calibrated mapping:

```python
percentile = 100 - target_afhp * 100
threshold  = policy.train_percentile_level(percentile)
```

The policy is then evaluated at this threshold. If the observed AFHP falls
inside `[bin_lo, bin_hi]`, the worker is done in a single evaluation.

The quality of this guess depends on how well the training calibration data
generalises to the test distribution:

| Policy type | Heuristic source |
|---|---|
| `ThresholdPolicy` / `OODPolicy` | Empirical percentile of per-episode max scores from training rollouts |
| `TimestepRandomPolicy` | Formula `1 - p^(1/L)` using mean training episode length `L` |
| `ExponentialHeuristicPolicy` | Formula `1 - p^(2/(L(L-1)))` using mean training episode length `L` |
| `WaitPolicy` | Empirical percentile of training episode lengths |

See `docs/percentile_calibration.md` for derivations.

## Step 2: Adaptive binary search over [0, 100]

If the initial guess misses the bin, the worker binary-searches over the full
percentile range `[0, 100]`, seeded from the initial guess:

```
# Initial guess told us which direction we need to go.
if afhp > bin_hi:        # AFHP too high, need higher threshold (higher pct)
    lo_pct = init_pct
    hi_pct = 100.0
else:                    # AFHP too low, need lower threshold (lower pct)
    lo_pct = 0.0
    hi_pct = init_pct

for each iteration:
    mid_pct   = (lo_pct + hi_pct) / 2
    threshold = train_percentile_level(mid_pct)
    → evaluate → observe afhp

    if afhp > bin_hi:   # still too high, raise floor
        lo_pct = mid_pct
    elif afhp < bin_lo: # too low, lower ceiling
        hi_pct = mid_pct
    else:
        break           # found it
```

The search terminates when the observed AFHP lands in the bin or
`search_depth_limit` iterations are reached (default 10).

### Why [0, 100] and not the narrow bracket

An earlier design constrained the search to the training-calibrated bracket
`[100 - bin_hi*100, 100 - bin_lo*100]`, reasoning that the training percentile
should map onto the correct AFHP range. This assumption breaks when:

1. **Distribution shift**: training and test score distributions differ, so the
   threshold range implied by the training calibration produces a different
   test-AFHP range that may not include the target bin.
2. **Discrete scores**: policies like `WaitPolicy` have integer thresholds.
   The calibrated bracket may contain no integer threshold that achieves the
   target AFHP.

Using `[0, 100]` means the search can always reach the correct threshold as
long as one exists. The initial guess is still used — it seeds the bracket so
the first midpoint is genuinely new information, not a repeat evaluation.

### Remaining limitations

**Threshold range is bounded by training data.** Percentile space `[0, 100]`
maps to a threshold range of `[train_percentile_level(0), train_percentile_level(100)]`,
i.e., the minimum and maximum scores seen in training. If test scores fall
outside this range, the corresponding AFHP values are unreachable by the
search. For example, if test episodes consistently produce higher scores than
any training episode, even the threshold `train_percentile_level(100)` may
not be high enough to drive AFHP to near zero.

The correct fix would be to search directly in threshold space and set bounds
based on the actual test score range. This is not currently implemented because
threshold space has no natural policy-agnostic scale (a max-logit value and a
reconstruction error are not comparable), making it hard to define a
policy-generic search procedure. In practice this limitation matters most under
large distribution shift between training and test.

**Discontinuous AFHP functions.** Some policies (notably `WaitPolicy`) have
discrete thresholds where a one-step change flips many episodes simultaneously.
If the AFHP function "jumps over" a bin entirely, no threshold achieves the
target AFHP and the search converges to the nearest jump point. In that case
the worker saves whatever result it found and logs a warning. These out-of-bin
points are visible in the final data as points that do not fall at their
`desired_percentile`.

## Example: `num_bins=20`, bin 6 (`level_afhp`)

```
bin_lo = 0.30,  bin_hi = 0.35,  target_afhp = 0.325
percentile bracket: [65, 70]

Initial guess:
  percentile = 100 - 32.5 = 67.5
  threshold  = train_percentile_level(67.5)
  → eval → afhp = 0.31   ✓ inside [0.30, 0.35] → done in 1 eval
```

If the initial guess had landed outside:

```
  → afhp = 0.28  (too low, need lower threshold → lower percentile)
  → expand bracket: lo_pct = 0.0,  hi_pct = 67.5
  mid_pct = 33.75
  threshold = train_percentile_level(33.75)
  → eval → afhp = 0.55  (too high)
  → lo_pct = 33.75,  hi_pct = 67.5
  mid_pct = 50.625
  → ...converges toward the right threshold
```

Because the bracket starts from `[0, 100]` rather than the narrow training
bracket, the search can reach any achievable AFHP value.

## Worst-case evaluation budget

Each bin uses at most `1 + search_depth_limit` evaluations. With 20 bins and
`search_depth_limit=10` the total budget is at most 220 evaluations. In
practice the heuristic is close enough that most bins need only 1–3 evaluations.

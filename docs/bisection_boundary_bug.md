# Bisection boundary-selection bug in `ImageSVDDRawThresholdSampler`

## 1. Bug explanation

### TL;DR

`ImageSVDDRawThresholdSampler._select_widest_fillable_interval` sorts
points by AFHP only. Within ties (e.g. all points with AFHP=1.0), Python's
stable sort preserves *insertion order*, which is unrelated to threshold
proximity. As a result, the "boundary pair" the sampler bisects can be
much wider than the tightest sandwich it has already discovered, and
its midpoint can collide with an already-evaluated threshold, triggering
a spurious `duplicate_threshold` early stop while a finer-grained
window is still unexplored.

### Location

`YRC/coverage/coverage_search.py`, function
`ImageSVDDRawThresholdSampler._select_widest_fillable_interval`:

```python
def _select_widest_fillable_interval(self):
    ordered = sorted(self.points, key=lambda point: point["afhp"])
    best_interval = None
    best_gap = -1.0
    for left, right in zip(ordered[:-1], ordered[1:]):
        gap = right["afhp"] - left["afhp"]
        ...
```

The bisection step in `_fill_afhp_bins` then takes
`midpoint = (left["threshold"] + right["threshold"]) / 2` and stops the
search if that midpoint hits a previously-seen threshold key.

### How it manifests

Reproduced on rnn smoke-test job `1132772` (commit `8ba74c6`), coinrun
svdd-image. After the raw sampler ran 21 evaluations, the points list
contained:

| AFHP | threshold                  | how it was produced     |
|------|---------------------------|-------------------------|
| 1.0  | 0.320000022649765         | id_threshold (eval 7)   |
| 1.0  | 0.320000022649765 (+1 ULP)| id_threshold_next (8)   |
| 1.0  | 0.32000003046226555       | bisect midpoint (21)    |
| 0.0  | 0.3200000382747661        | bisect midpoint (20)    |
| 0.0  | 0.32000005, ..., 0.32003202 | earlier expansion / bisects |

The tightest sandwich across the AFHP boundary is
`(0.32000003046226555, 0.3200000382747661]`, width 7.81e-9. Inside that
window AFHP is unknown.

Because the sort key is `point["afhp"]` only, the AFHP=1 cluster is
returned in insertion order: eval 7, eval 8, eval 21. The boundary
pair the sampler selects is therefore

```
left  = eval 20 (threshold = 0.3200000382747661, afhp = 0)
right = eval 7  (threshold = 0.320000022649765,  afhp = 1)
```

Its midpoint is

```
(0.3200000382747661 + 0.320000022649765) / 2 = 0.32000003046226555
```

which is exactly the threshold of **eval 21**. `_threshold_key(midpoint)`
matches `seen_thresholds`, the interval gets recorded as
`duplicate_threshold`, and `_fill_afhp_bins` exits with
`early_stop_reason="unfillable_afhp_intervals"`.

If the sampler had instead picked the closest pair across the AFHP
boundary, it would have bisected eval 21 vs eval 20, midpoint
`0.32000003436851... ` , a never-seen threshold. The bisection could have
continued narrowing the actual transition window.

### Why this is a real bug, not just an efficiency issue

The current behavior terminates with `coverage_percentage = 40%`
(2 of 5 bins filled) and silently claims there is no fillable
intermediate AFHP region. The actual reason for early stop is *internal
sampler bookkeeping*, not a property of the AFHP curve. A correctly
implemented sampler might or might not find intermediate AFHP in the
7.8e-9 window, but the current code never gets the chance to look.

This matters whenever:
1. There are multiple points with tied AFHP near the boundary
   (always true once a few bisection steps have happened on a near-step
   AFHP curve), and
2. The closest cross-boundary pair was produced after earlier, wider
   pairs.

Both conditions hold for any image-SVDD checkpoint where the AFHP curve
is a step function in raw-threshold space, which appears to be common
based on the smoke test.

### Possible fixes

#### Option A: sort within AFHP ties so boundary-adjacent points end up adjacent

Use `(afhp, -threshold)` as the sort key (ascending). Within an AFHP=v
cluster this gives points in *descending* threshold order, so the last
element of one AFHP cluster has the smallest threshold among ties, and
the first element of the next AFHP cluster has the largest threshold
among ties. For our case this would place eval 21 (largest threshold in
AFHP=1 cluster) immediately adjacent to eval 20 (smallest threshold in
AFHP=0 cluster) across the boundary.

Caveat: this assumes the AFHP curve is monotonically decreasing in
threshold. If not, the heuristic may pick the wrong pair. For the
current image-SVDD path the assumption holds.

#### Option B: select boundary points explicitly per AFHP gap

Instead of relying on a global sort, iterate over distinct AFHP values
(or AFHP buckets), and for each gap pick:
- `left`  = the point in the lower-AFHP cluster with the **largest**
  threshold that is still **below** any threshold in the higher-AFHP
  cluster (and similarly `right` with smallest above).

This is more code but does not bake in monotonicity assumptions.

#### Option C: deduplicate AFHP-equal points first

Before sorting, collapse each AFHP-cluster to its threshold-extreme
representatives only (smallest threshold in each AFHP=v cluster as one
representative, largest threshold as another). This avoids accidentally
introducing wider pairs from interior cluster points.

### Suggested patch

Implement Option A as the smallest change. Add a regression test that
constructs a raw sampler with the eval-21 / eval-20 / eval-7 / eval-8
points pre-loaded and confirms the next bisection target is
`midpoint(eval 21, eval 20)` rather than `midpoint(eval 20, eval 7)`.

---

## 2. Running the smoke test on rnn

The smoke test reproduces the exact failure mode the sampler is
designed to handle (degenerate calibration scores, near-step AFHP
curve on eval). It is the cheapest end-to-end check that any sampler
patch behaves correctly.

### Prerequisites

- SSH access to `rnn` configured as host alias `rnn`.
- The branch under test must be pushed to `origin/ood`. The smoke test
  pulls from `origin/ood` on rnn before submitting.
- Conda env `ood-stable` must exist at
  `/nas/ucb/czempin/anaconda3/envs/ood-stable` (frozen experiment env).
- The repo on rnn is at `/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork`.

### Pulling the latest commit

```bash
ssh rnn 'cd /nas/ucb/czempin/code/goal-misgen/yrc-bench-fork \
  && git fetch origin ood \
  && git pull --ff-only \
  && git log --oneline -3'
```

Confirm the expected commit SHA is at the top.

### Submitting the job

```bash
ssh rnn 'cd /nas/ucb/czempin/code/goal-misgen/yrc-bench-fork \
  && source /nas/ucb/czempin/anaconda3/etc/profile.d/conda.sh \
  && conda activate ood-stable \
  && python scripts/run_eval.py \
       --env coinrun \
       --method svdd-image \
       --prefix debug-image-svdd-threshold \
       --exp-ids 0 \
       --num-levels 16 \
       --coverage-fraction 0.2 \
       --calibration-levels 64 \
       --video-episodes 0 \
       --video-filter all \
       --cp-rolling-average none \
       --video-logging-mode folder \
       --video-filter-mode any'
```

The script prints a line like
`Submitted coinrun_svdd-image_exp0: Submitted batch job <JOBID>`.
Record `JOBID`.

To preview the sbatch script without submitting, add `--dry-run`.

### Log locations

For job `JOBID`, the two SLURM files are at

```
/nas/ucb/czempin/data/goal-misgen/slurm-logs/default/debug-image-svdd-threshold/YYYY-MM-DD/coinrun_svdd-image_exp0_<JOBID>.err
/nas/ucb/czempin/data/goal-misgen/slurm-logs/default/debug-image-svdd-threshold/YYYY-MM-DD/coinrun_svdd-image_exp0_<JOBID>.out
```

The `.err` file holds the per-run log (everything from `logging.info`,
including the eval tracker's per-iteration lines and the sampler
dispatch / probe diagnostics). The `.out` file holds only the
launcher shell's stdout, which is mostly empty for a still-running job
because Python stdout is block-buffered.

### Signals to watch for in `.err`

In rough chronological order:

1. `Generated OOD calibration scores: <N> step scores, <M> episode max scores`
   - Calibration finished. Expect `M = 64` with the default settings.

2. `Sampler dispatch: policy_type=OODPolicy, clf_name=DeepSVDD, feature_type=obs, image_svdd_degenerate_strategy='expand_above_id', image_svdd_diagnostics={...}`
   - The dispatch site logged the diagnostics. If
     `image_svdd_degenerate_strategy=None` or the diagnostics show
     `is_image_svdd=False`, the probe will be skipped. That itself is
     a bug if you expected the probe to run.

3. `Sampler dispatch: selecting ImageSVDDProbeSampler (...)`
   - Probe path was selected.

4. `Image SVDD percentile probe starting with percentiles=(0.25, 0.5, 0.75, 0.9)`
   - Probe began. The next 6 evals are the probe (lower extreme,
     upper extreme, then four interior percentiles).

5. `Image SVDD percentile search collapsed; switching to raw threshold expansion above ID threshold <id_threshold>. Reason=<reason>`
   - Probe decided to switch. The reason is one of
     `degenerate_calibration`, `duplicate_finite_thresholds`,
     `all_finite_probes_high_afhp`.

6. `[Eval N] threshold=..., step_afhp=..., level_afhp=..., performance=...`
   - Per-evaluation line from `EvalStepTracker.log_eval`. Threshold is
     printed at `%14.8g` precision.

7. `Sampling info: {...}` near the end
   - Final diagnostics. Look for `early_stop_reason`,
     `coverage_percentage`, `unfillable_afhp_intervals`,
     `bins_filled`, `total_bins`.

### Tailing in real time

```bash
ssh rnn 'tail -F /nas/ucb/czempin/data/goal-misgen/slurm-logs/default/debug-image-svdd-threshold/YYYY-MM-DD/coinrun_svdd-image_exp0_<JOBID>.err'
```

### Cancelling

```bash
ssh rnn 'scancel <JOBID>'
```

### Typical timing on rnn (`ood-stable` env, single GPU)

- Calibration: about 2 minutes for 64 episodes.
- Per evaluation (any threshold): about 1 to 2 minutes.
- The probe burns 6 evals before the dispatch decision, so the
  earliest signal that the probe selected raw expansion arrives
  roughly 10 minutes after submission.

---

## 3. Inspecting raw SVDD scores directly

Before investing more sampler work, find out whether intermediate AFHP
is even achievable on this checkpoint. If the SVDD produces literally
one float64 score for every input, no bisection strategy can produce
non-trivial AFHP.

### What to check

1. **Per-frame score distribution on the calibration env.** Calibration
   uses `random_percent=0` (ID-themed only). Verify the per-step (not
   per-episode-max) score distribution: range, unique-value count,
   approximate histogram. If even per-frame scores are a delta, the
   SVDD has fully collapsed.

2. **Per-frame score distribution on `random_percent=100` levels.**
   Build a coinrun env with `random_percent=100` and pass a few
   thousand frames through `_compute_scores`. Compare the resulting
   distribution to the ID-only distribution from step 1. If they
   overlap with no gap, the SVDD is not separating themes at all.
   If they are offset by some amount on the order of 1e-8 to 1e-5,
   that is the actual signal the sampler is trying to surface.

3. **Per-episode-max score distribution on eval (`random_percent=50`).**
   Compare against the calibration episode-max distribution. If
   episode-max collapses to a single float64 value even though
   per-frame scores have spread, the bottleneck is the aggregation
   choice (`max` over a 50-200 frame episode), not the model or the
   sampler.

### Suggested script outline

Place at `scripts/inspect_image_svdd_scores.py`. Use the project's
preferred Pathlib idiom for paths.

```python
"""Probe the per-frame and per-episode-max score distributions of a
trained image-SVDD checkpoint, to diagnose whether AFHP-axis spread is
recoverable on a given checkpoint."""
from pathlib import Path
import numpy as np
import torch

import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
import flags

# Reuse the same flags / config loader that eval_afhp.py uses, so
# checkpoint, feature_type, etc. are configured exactly the same way.
args = flags.make()
args.eval_mode = True
config = config_utils.load(args.config, flags=args)

envs = env_factory.make(config, ...)
policy = policy_factory.make(config, envs["train"])
policy.load_model(Path(config.experiment_dir, config.file_name))

# 1. Roll out N frames on an ID-only env (random_percent=0).
# 2. Roll out N frames on an OOD-only env (random_percent=100).
# 3. Roll out N frames on the actual eval env (random_percent=50).
# For each: record per-frame scores AND per-episode-max scores.
# Save raw arrays as npz so they can be inspected separately.
```

Key numbers to print:

- `unique_count(per_frame_scores)` using
  `np.unique` with a tolerance that matches the sampler's
  (`DEFAULT_SCORE_TOLERANCE_ABS = 1e-8`,
  `DEFAULT_SCORE_TOLERANCE_REL = 1e-6`).
- `np.min, np.max, np.std` on the per-frame array.
- Same on the per-episode-max array.
- A coarse histogram via `np.histogram` with say 50 bins between
  the global min and max.

### What the answers tell us

| ID-frame spread | OOD-frame spread | Outcome                                                |
|-----------------|-----------------|--------------------------------------------------------|
| zero            | zero            | Model fully collapsed. No sampler change can help.     |
| zero            | non-zero        | Model separates ID from OOD on a per-frame basis. The sampler bug in section 1 is worth fixing. |
| non-zero        | non-zero, overlap | Model has some sensitivity but does not separate themes. Aggregation change (mean / fraction-above-threshold) may help. |
| non-zero        | non-zero, no overlap | Model works at per-frame level. Episode-max aggregation is the only thing collapsing the signal. Switching aggregation is the fix. |

### Running on rnn

Copy or rsync the script into the rnn checkout, then

```bash
ssh rnn 'cd /nas/ucb/czempin/code/goal-misgen/yrc-bench-fork \
  && source /nas/ucb/czempin/anaconda3/etc/profile.d/conda.sh \
  && conda activate ood-stable \
  && python scripts/inspect_image_svdd_scores.py \
       -c configs/eval/coinrun/image_svdd.yaml \
       -en coinrun \
       -sim <weak.pth> -weak <weak.pth> -strong <strong.pth> \
       -f_n /nas/ucb/czempin/data/goal-misgen/trained_svdd/neurips04/svdd_coinrun_image_exp0/trained.joblib \
       -level_seeds_file /nas/ucb/czempin/data/goal-misgen/seeds/icml/0.json'
```

GPU is not strictly required but is much faster. Run for a few thousand
frames; one minute or two is enough to get a clean distribution.

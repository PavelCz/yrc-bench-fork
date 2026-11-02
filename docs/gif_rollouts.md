# GIF rollouts for the project website

K&C (Keys and Chests) uses the Procgen environment `heist_afh` and the
configs in `configs/eval/heist/`. The website's existing format is an
upscaled 64×64 agent observation above the 512×512 human observation.
Coordination clips also have a score or action bar; red means expert control.

## Where to run

Small recording jobs can run locally on CPU in the `ood` environment.
Activate it and pass `-d -1`. Limit CPU math threads, for example with
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. No GPU or W&B account is needed.

On 22 October 2026, the K&C checkpoints were found on **Snoopy**, under
`/scr/pavel/data/goal-misgen`, including the 400M experts in
`policy/heist400`. The standard CARC data root
`/project2/biyik_1165/czempin/data/goal-misgen/policy` contained Coinrun and
Maze, but no K&C checkpoints. Do not assume that changing the RNN prefix
to the CARC prefix makes K&C inputs available. Copy the required checkpoints
first. For CARC's Apptainer runtime, see [container_workflow.md](container_workflow.md).

## Standalone novice or expert

Use `eval_policy.py`, passing the relevant acting-policy checkpoint:

```bash
python eval_policy.py -c configs/eval/heist/max_prob.yaml \
  -n kc_novice -experiment_group kc-website -en heist_afh \
  --model_file /path/to/novice.pth \
  -level_seeds_file /path/to/selected-levels.json \
  -eval_split test -num_rollouts 4 -num_envs 1 \
  -video_episodes_to_collect 4 -video_logging_mode folder \
  -video_output_folder /absolute/path/to/videos/novice \
  -greedy false -wandb_mode disabled -d -1
```

The selected-levels JSON uses the normal seed-file format, with the desired
levels in `seeds.ood_eval`. Keep fixed seeds and the full results JSON so the
captions can be checked against actual completion, keys, and chests.
Positive return alone does not establish that a K&C level was completed.

To search specifically for novice OOD timeouts, use
`collect_ood_failure_videos.py` with the same flags. Here `-num_rollouts` is
the search budget and `-video_episodes_to_collect` is the desired number of
matching failures. This script writes an `ood_timeout_videos.json` manifest.

## Coordination at a chosen threshold

`record_coordination_videos.py` uses the existing evaluator and video
renderer, without calibration or an adaptive AFHP sweep. Copy the desired
evaluation YAML and set `evaluation.video_threshold`. For example, for
`ThresholdPolicy` with `metric: max_prob`, use `video_threshold: 0.5`.
The current score is `1 - max(action probability)`, and expert control is
requested when the score exceeds the threshold. As in `scripts/run_eval.py`
with `-defer_to_oracle`, the recorder keeps the expert in control for the
rest of the episode after the first query. Without that setting, the raw
threshold policy can return control to the novice when its score falls.

```bash
python record_coordination_videos.py -c /path/to/video-config.yaml \
  -n kc_max_prob -experiment_group kc-website -en heist_afh \
  -sim /path/to/novice.pth -weak /path/to/novice.pth \
  -strong /path/to/expert400m.pth \
  -level_seeds_file /path/to/selected-levels.json \
  -num_rollouts 4 -num_envs 1 \
  -defer_to_oracle \
  -video_output_folder /absolute/path/to/videos/max-prob \
  -wandb_mode disabled -d -1
```

For a wait-then-ask example, set `coord_policy.cls: WaitPolicy` and
`evaluation.video_threshold: 40` to switch to the expert after 40 steps.
Thresholds chosen for illustrative clips are not calibrated AFHP budgets.
Use `eval_afhp.py` and the full seed splits for performance curves.

The recorder saves the resolved config, per-level results, and a mapping
from renderer video indices to level seeds in its evaluation directory.
It requires `-num_envs 1` so selected levels are recorded in order, and adds
one unused guard seed so Procgen's last reset preserves terminal counters.
Raw GIFs and JSON results go under `experiments/`, which is ignored by Git.

## Website encoding

The renderer writes full-resolution debugging GIFs that can be large.
Re-encode selected clips with a shared palette while preserving timing,
resolution, and infinite looping:

```bash
ffmpeg -i input.gif \
  -filter_complex '[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle' \
  -loop 0 output.gif
```

Keep source paths, checkpoints, level seeds, thresholds, and outcome counts
in an accompanying manifest. Inspect the first, middle, and last frames
before adding a selected clip to the website's `static/videos/` directory.

# Level Seed Splits

`generate_level_seeds.py` writes a JSON file with four semantic seed splits:
`policy_train`, `ood_train`, `validation`, and `ood_eval`. These splits are
non-overlapping. In paper mode, `policy_train` is the sequential range
`[0, 100000)`, while the other splits are shuffled seeds from the range starting
at `100000`.

The semantic seed split names are not always the same as YRC environment split
names. Callers explicitly map seed-file splits to environment splits. For
example, `eval_afhp.py` maps `validation` to the env split `cal`, and maps
`ood_eval` to the env split `test`.

## Split Roles

| Seed split | Used for | Main consumers |
|---|---|---|
| `policy_train` | Train the weak, simulated weak, strong, and ensemble acting policies. | `lib/train-procgen-pytorch/train.py`, via `scripts/train_policies.sh` and `scripts/train_ensemble_policies.sh` |
| `ood_train` | Collect rollout artifacts used to train OOD detectors such as Deep SVDD. | `gather_rollouts.py`, via `scripts/run_gather_rollouts.py` |
| `validation` | Held-out validation and calibration levels. Used for acting-policy validation and AFHP threshold/percentile calibration. | `lib/train-procgen-pytorch/train.py`; `eval_afhp.py` as env split `cal` |
| `ood_eval` | Final held-out evaluation levels. | `eval_afhp.py` as env split `test`; `eval_policy.py`; `eval_strong_on_help.py` |

## Pipeline Usage

1. Generate level seed files with `generate_level_seeds.py`.
2. Train acting policies on `policy_train`.
3. Validate acting policies on `validation`.
4. Gather OOD training rollouts on `ood_train`.
5. Train Deep SVDD or other OOD detectors from the `ood_train` rollout artifact.
6. Calibrate AFHP thresholds or percentiles on `validation`.
7. Evaluate final policies on `ood_eval`.

`gather_rollouts.py` uses only `ood_train`. When multiple rollout dataset sizes
are needed, `scripts/run_gather_rollouts.py --num-levels 64 128 all` selects
prefixes of `ood_train`. SVDD training can also load the largest available
rollout artifact and restrict it with `--rollout-max-levels`, avoiding duplicate
collection for smaller dataset-size runs.

## Additional OOD Training Seeds

For larger OOD detector datasets, generate a separate OOD-train-only seed file
rather than editing the canonical level seed files:

```bash
python scripts/generate_extra_ood_train_seeds.py \
    --existing-level-seeds /path/to/cluster1/data/goal-misgen/seeds/icml/0.json \
    --ood-train 1024 \
    --base-seed 6033 \
    --name extra_ood_train_1024 \
    -o /path/to/cluster1/data/goal-misgen/seeds/extra_ood_train_1024/0.json
```

The generated file has the standard `seeds` object, but only `ood_train` is
non-empty. It excludes every seed from every split in the listed existing files,
and writes the set name to `metadata.name`. Use it for rollout collection with:

```bash
python scripts/run_gather_rollouts.py \
    --env coinrun \
    --prefix rollouts-neurips \
    --exp-ids 0 \
    --server cluster1 \
    --num-levels all \
    --level-seeds-file /path/to/cluster1/data/goal-misgen/seeds/extra_ood_train_1024/0.json
```

## Naming Notes

- `validation` is a seed-file split.
- `val_sim` and `val_true` are YRC environment split names used for coordination
  policy validation/checkpoint naming. They do not automatically refer to the
  seed-file `validation` split.
- `cal` is an environment split created for AFHP calibration. In the current
  evaluation path, it receives the seed-file `validation` split.
- `test` is an environment split. In fixed-seed AFHP evaluation, it receives the
  seed-file `ood_eval` split.

As of the current code, `train_svdd.py` trains from rollout artifacts gathered on
`ood_train`; it does not automatically consume the seed-file `validation` split.
If training-time SVDD validation curves are added, the `validation` split is the
appropriate held-out seed split to use for them. A subset of `ood_train` can be
used for train-set diagnostics, but those results should be labeled as training
curves rather than validation curves.

# Snoopy migration

On 15 October 2026 the artifacts needed to keep training and evaluating
without RNN were copied to snoopy. The destination is the snoopy entry in
`scripts/common.py` `SERVER_PATHS`:

```text
/scr/pavel/data/goal-misgen
```

`scripts/migrate_to_snoopy.py --help` is the inventory of what was copied.
Most launchers still default to the RNN layout (`--server chai`,
`/nas/ucb/czempin/...`). A job on snoopy has to be pointed at the tree
below. Do not send it back to the RNN paths.

## Change the launchers

Python launchers take `--server`. Pass `snoopy`.

| Script | Current default |
| --- | --- |
| `scripts/run_eval.py` | `chai` |
| `scripts/run_svdd_train.py` | `chai` |
| `scripts/run_eval_policy.py` | `chai` |
| `scripts/run_eval_strong_on_help.py` | `chai` |
| `scripts/sync_eval_artifacts.py` | `chai` |
| `scripts/run_gather_rollouts.py` | `snoopy` |

These shell scripts have no `--server` flag. Their conda base, seed
directory, log directory, and checkpoint directory are still absolute RNN
paths:

- `scripts/train_policies.sh` (`CONDA_BASE`, `LEVEL_SEEDS_FOLDER`, `LOG_DIR`, `CHECKPOINT_BASE`)
- `scripts/train_ensemble_policies.sh` (`CONDA_BASE`, `LEVEL_SEEDS_FOLDER`, `LOG_DIR`)

`scripts/common.py` `build_chai_cache_env_block` and `CHAI_CONDA_BASE`
apply only when `server == "chai"`. The migration did not copy
`/nas/ucb/czempin/anaconda3`. `docs/rnn_smoke_tests.md` is the RNN workflow;
do not follow it on snoopy.

Figure scripts default to the laptop directory
`/home/pavel/data/goal-misgen/icml-evals`
(`DEFAULT_EVAL_DIR` in `analyzing/plot_main_figures.py` and the other
`analyzing/plot_*.py` figure scripts). On snoopy the curves are under
`experiments/evals` in the data root above.

## Paths that are not a prefix swap

Replacing `/nas/ucb/czempin/data/goal-misgen` with
`/scr/pavel/data/goal-misgen` is right for files that were copied, and
wrong for the lookups below.

SVDD models are not all under `trained_svdd/neurips05`.
`scripts/run_eval.py` `EVAL_DEFAULTS["svdd_prefix"]` is `neurips05`, which
is the CoinRun prefix only.

| Environment | Directory under `trained_svdd/` |
| --- | --- |
| CoinRun | `neurips05` |
| Maze | `tmlr-maze-svdd-fix-model` |
| Heist | `tmlr-heist01` |

There is no Maze model under `trained_svdd/neurips05` on snoopy. Image and
latent, seeds 0–7, are the models that were copied.

Rollout gathers are not all under `rollouts/neurips03`:

| Gather | Relative path |
| --- | --- |
| CoinRun, seeds 0–7 | `rollouts/neurips03/coinrun/gather_coinrun_expN` |
| Heist, seeds 0–7 | `rollouts/tmlr-heist01/heist/gather_heist_expN` |
| Maze, seeds 0–3 | `rollouts/tmlr-maze-svdd-fix/maze/gather_maze_expN` |
| Maze, seeds 4–7 | `rollouts/neurips03/maze/gather_maze_expN` |

Seed files are `seeds/icml/{0..7}.json`,
`seeds/icml/ensemble-{0..7}_{0..3}.json`, and
`seeds/neurips_extra_ood_train_1024/{0..7}.json`. `SERVER_PATHS` `seeds_base`
is only `seeds/icml`.

Ensemble members are under `policy/icml/<env_folder>/ensembles/`. The RNN
checkout copies in `lib/train-procgen-pytorch/logs/train/` were not
migrated. Checkpoint lookup on snoopy has to use the policy tree.

Heist 400M experts are under `policy/heist400`. Maze robust400 experts are
under `policy/neurips/maze_afh_random_start`. The non-robust Maze 50%
experts `policy/icml/maze_afh/icml2_maze_exp{5,6,7}_50p` were not on RNN
and are not on snoopy. The robust400 experts for those seeds are present.

Winning eval `.npz` files are in
`/scr/pavel/data/goal-misgen/experiments/evals`. That is one directory,
not the three RNN trees they were read from (the fork checkout, the
`yrc-bench-afhp-range-retry` checkout, and the RNN data-tree evals).
Only the newest run per environment, method, and seed was copied, for the
prefixes the figure scripts still read, seeds 0–7. Older reruns and
prefixes those scripts no longer read were left on RNN.

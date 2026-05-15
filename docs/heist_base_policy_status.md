# Heist Base Policy Training — Status Log

Resumable status for the heist base-policy work on the `heist` branch. If this
session is lost, any future Claude Code invocation (or human) should be able to
read this file plus the plan it points to and pick up exactly where things were
left off.

## Goal

Train **base acting policies** for heist on chai (SSH host: `rnn`):
- **Weak agent**: PPO trained at `random_percent=0` on `heist_afh` (ID only —
  many_chests behavior, 1:2 keys:chests). Following Langosco et al, this agent
  should display goal misgeneralization on the OOD variant.
- **Strong/expert agent**: PPO trained at `random_percent=50` (50/50 mix of
  many_chests and many_keys behaviors).
- Single seed: **exp_id=0** (default heist seed = 1111).
- Conda env on rnn: **`ood`** (NOT `ood-stable` — the latter is frozen for the
  parallel OOD work in `/home/pavel/code/goal-misgen/yrc-bench-fork`).

These checkpoints unblock a later, separate plan to extend `scripts/run_eval.py`
to heist (ID = `heist_aisc_many_chests`, OOD = `heist_aisc_many_keys`).

## Repo layout on rnn

The rnn account is `czempin` (the user's local machine uses `pavel`; on rnn the
equivalent paths are `/home/czempin/...`).

- **Heist clone** (this branch): `/home/czempin/code/goal-misgen/heist-yrc-bench-fork`
- **OOD clone** (parallel, do not touch): `/home/czempin/code/goal-misgen/yrc-bench-fork` (currently on branch `ood`)

All commands below assume the heist clone.

## Pointer back to plan

See `/home/pavel/.claude/plans/the-current-eval-setup-glimmering-thunder.md`
(local to the originating Claude Code session — replicate or paste into the
status as needed if the plan file isn't accessible from a fresh machine).

## Step checklist

### Step 0 — Status log
- [x] Created this file.

### Step 1 — Local pre-flight
- [x] Confirmed `heist_afh` is compiled into the local
  `lib/procgen/procgen/.build/relwithdebinfo/libenv.so` (`strings` shows
  `heist_afh`, `_GLOBAL__sub_I_heist_afh.cpp`).
- [x] Skimmed `lib/procgen/docs/RANDOM_PERCENT_HEIST_IMPLEMENTATION_PLAN.md` —
  the planned implementation is what's in `heist_afh.cpp` today.
- **Note**: debug info in the local `libenv.so` embeds an absolute source path
  pointing to the OLD clone (`/home/pavel/code/p-goal-misgen/yrc-bench-fork/...`),
  meaning that `.so` was originally built from there. Doesn't block anything —
  the rnn install is what actually matters and gets verified in Step 2.

### Step 2 — rnn pre-flight (SSH)
- [x] Heist clone exists at `/home/czempin/code/goal-misgen/heist-yrc-bench-fork`
  on branch `heist`, up to date with `origin/heist` (HEAD `4adbab2`). User
  created the clone fresh from GitHub. Submodules (`Minigrid`, `cliport`,
  `pytorch_vae`) are uninitialized — not needed since this fork is
  Procgen-only and `lib/procgen` / `lib/train-procgen-pytorch` are vendored
  in the main repo, not submoduled.
- [x] `ood` env on rnn exposes `heist_afh`. The first `from procgen import
  ProcgenGym3Env; ProcgenGym3Env(env_name='heist_afh', ...)` triggered a
  procgen auto-rebuild (`building procgen...done`) and returned
  `Dict(rgb=D256[64,64,3]) D15[]` — correct obs/action spaces. No manual
  rebuild required.
  - To re-run: `ssh rnn "source /nas/ucb/czempin/anaconda3/etc/profile.d/conda.sh && conda activate ood && python -c \"from procgen import ProcgenGym3Env; print(ProcgenGym3Env(num=1, env_name='heist_afh', random_percent=0, distribution_mode='hard').ob_space)\""`
- [x] Existing maze layout confirmed:
  `/nas/ucb/czempin/data/goal-misgen/policy/icml/maze_afh/icml2_maze_exp0_0p/2026-01-13__15-01-05__seed_1080/model_200015872.pth`
  (plus 10 intermediate checkpoints). This is exactly the layout
  `train_policies.sh -e heist -x 0` will produce after the `--logdir_base`
  edits.

### Step 3 — Smoke training run on rnn (100k timesteps)
- [x] Submitted job 1134987 via
  `ssh rnn "cd /home/czempin/code/goal-misgen/heist-yrc-bench-fork && scripts/train_policies.sh -e heist -x 0 --random-percent 0 --num-timesteps 100000 --days 1"`.
  **State**: COMPLETED in 60s (setup 4.78s, training 49s, ~2038 steps/s).
- [x] Checkpoints written to
  `/nas/ucb/czempin/data/goal-misgen/policy/icml/heist_afh/icml2_heist_exp0_0p/2026-05-14__01-36-03__seed_1111/`:
  `model_65536.pth`, `model_131072.pth`.
- [x] `get_checkpoints('heist', 0, '/nas/ucb/.../policy/icml')` resolves:
  - `weak` and `sim` → `.../model_131072.pth`
  - `strong` → `.../icml2_heist_exp0_50p/NOT_FOUND` (expected; strong not yet trained).
  Warning fires that 131072 ≠ EXPECTED_TIMESTEPS (200015872) — expected for a smoke run.
- Wandb is not enabled by default in `train_policies.sh` (no `--use_wandb`); skipped.

**Failures along the way (resolved):**
- Job 1134953 (first attempt): FAILED in 1s with `conda: not found`. The
  `sbatch --wrap` script runs under `/bin/sh` which doesn't load
  `~/.bashrc`. When sbatch is invoked from a non-interactive ssh, conda
  isn't on PATH. Fixed in commit `ca05959` by sourcing
  `${CONDA_BASE}/etc/profile.d/conda.sh` in the wrap.
- Job 1134986 (second attempt): FAILED in 1s with `source: not found`.
  `/bin/sh` (Dash) doesn't have the bash-only `source` builtin. Fixed in
  commit `8d32279` by switching to POSIX `.` (dot).
- Both failures are now self-contained and won't recur regardless of how
  `train_policies.sh` is invoked.

### Step 4 — Full 200M training (weak + strong)
- [x] **Weak** submitted: SLURM job `1134988`,
  `scripts/train_policies.sh -e heist -x 0 --random-percent 0`
  (200M timesteps, 3-day wall). Log:
  `/nas/ucb/czempin/data/goal-misgen/logs/train_policies/icml2_heist_exp0_0p_1134988.out`.
  Expected output dir:
  `/nas/ucb/czempin/data/goal-misgen/policy/icml/heist_afh/icml2_heist_exp0_0p/<timestamp>__seed_1111/model_200015872.pth`.
- [x] **Strong** submitted: SLURM job `1134989`,
  `scripts/train_policies.sh -e heist -x 0 --random-percent 50`
  (same params). Log:
  `/nas/ucb/czempin/data/goal-misgen/logs/train_policies/icml2_heist_exp0_50p_1134989.out`.
  Expected output dir:
  `/nas/ucb/czempin/data/goal-misgen/policy/icml/heist_afh/icml2_heist_exp0_50p/<timestamp>__seed_1111/model_200015872.pth`.

Estimated wall time per job: ~27h pure training (2038 steps/s measured
in the smoke run × 200M steps), plus ~5–10s setup. Watcher disabled per
user preference; resume Step 5 when both jobs leave the queue.

To check progress at any time:
```bash
ssh rnn "squeue -u czempin -j 1134988,1134989 -o '%i %T %M %R'"
ssh rnn "sacct -j 1134988,1134989 --format=JobID,State,ExitCode,Elapsed -P | head -10"
```

### Step 5 — Post-training verification
- [x] **Both jobs COMPLETED.** Weak (1134988): exit 0:0, 20h 55m, throughput
  2656 steps/s, final val_mean_reward 3.20. Strong (1134989): exit 0:0,
  20h 51m, throughput 2664 steps/s, final val_mean_reward 3.35.
- [x] `get_checkpoints('heist', 0, '/nas/ucb/.../policy/icml')` returns:
  ```
  sim/weak: .../icml2_heist_exp0_0p/2026-05-14__01-40-04__seed_1111/model_200015872.pth
  strong:   .../icml2_heist_exp0_50p/2026-05-13__18-52-39__seed_1111/model_200015872.pth
  ```
  No `NOT_FOUND` markers.
- [x] `find_best_model_checkpoint` resolved `model_200015872.pth` for both;
  matches `EXPECTED_TIMESTEPS` exactly (no mismatch warning). Each run
  produced 10 intermediate checkpoints (model_20054016.pth through
  model_180027392.pth) plus the final.
- [x] **Behavioral rollout sanity check** (sbatch job 1137495, ~7 min wall):
  100-episode test rollouts of each agent on `heist_afh` configured for
  ID (random_percent=0, many_chests behavior) vs OOD (random_percent=100,
  many_keys behavior), via `eval_policy.py` against the existing
  `configs/eval/heist/timestep_random_many_{chests,keys}.yaml` configs.

  | Agent | many_chests (ID) | many_keys (OOD) | Δ |
  |---|---|---|---|
  | weak (rp=0)   | 3.51 ± 1.51 | 2.96 ± 1.34 | −16% |
  | strong (rp=50)| 3.38 ± 1.50 | 3.40 ± 1.46 | ~0% |

  Strong agent generalizes (parity across both distributions). Weak
  degrades on OOD vs ID and underperforms strong on OOD — the
  qualitative split that downstream coordination eval depends on. The
  ~16% weak degradation is modest rather than catastrophic, consistent
  with heist's OOD variant still rewarding the same key-grab + chest-open
  loop just with an extra unused key.

  Two pre-existing issues found and worked around (not fixed):
  - The heist eval YAMLs (`configs/eval/heist/*.yaml`) lack a top-level
    `name:` field that other configs (e.g. `procgen_threshold.yaml`) carry.
    Workaround: pass `-n <name>` on the CLI. Worth adding `name:` to the
    YAML files when the eval-side plan touches them.
  - Compute nodes mount `/nas/ucb/czempin/...` but not `/home/czempin/...`.
    Helper scripts that run inside an sbatch job must reference the NAS
    path. `train_policies.sh` is fine because it uses `realpath` to
    resolve the symlink before baking into the wrap.

  Sanity-eval helper script: `/tmp/heist_sanity_eval.sh` on rnn (one-off,
  not committed). Per-eval JSON results saved under
  `experiments/evals/sanity_*/`.

**Housekeeping note:** the smoke-run dir
`.../icml2_heist_exp0_0p/2026-05-14__01-36-03__seed_1111/` (containing
only `model_65536.pth` + `model_131072.pth`) coexists with the full run
under the same `icml2_heist_exp0_0p` parent. `find_newest_timestamp_dir`
correctly picks the newer full-run dir but emits a "Multiple timestamp
dirs" warning. Safe to delete the smoke dir whenever convenient.

## Code changes already landed on this branch

Both bundled in commit:
- `lib/train-procgen-pytorch/train.py`: added `--logdir_base` CLI arg
  (default `logs/train`, backward-compatible). Used in place of the previous
  hardcoded `os.path.join("logs", "train", ...)` at line 345.
- `scripts/train_policies.sh`:
  - `CONDA_ENV` switched from `"ood-stable"` to `"ood"` (heist work uses the
    experimental env).
  - Added `CHECKPOINT_BASE="/nas/ucb/czempin/data/goal-misgen/policy/icml"`.
  - Pass `--logdir_base $CHECKPOINT_BASE` to `train.py` so checkpoints land
    where `scripts/common.py::get_checkpoints` looks for them — no manual
    file-move step required after training.

## Open questions / blockers

_(none yet — append as discovered)_

## Out of scope (deferred)

- Adding `"heist"` to `ENVS` in `scripts/common.py` and `EVAL_ENVS` in
  `scripts/run_eval.py` (handled in the eval-side plan).
- Ensemble member training (`scripts/train_ensemble_policies.sh`).
- Multi-seed (exp_ids 1–4).
- Heist-specific level seed files.

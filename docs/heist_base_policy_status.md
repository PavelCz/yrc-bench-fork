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

- **Heist clone** (this branch): `/home/pavel/code/goal-misgen/heist-yrc-bench-fork`
- **OOD-stable clone** (parallel, do not touch): `/home/pavel/code/goal-misgen/yrc-bench-fork`

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
- [ ] `ssh rnn "cd /home/pavel/code/goal-misgen/heist-yrc-bench-fork && git fetch && git checkout heist && git pull"`
- [ ] Verify `ood` env on rnn exposes `heist_afh`:
  ```bash
  ssh rnn "conda run -n ood python -c \"from procgen import ProcgenGym3Env; e = ProcgenGym3Env(num=1, env_name='heist_afh', random_percent=0, distribution_mode='hard'); print(e.ob_space, e.ac_space)\""
  ```
  - If this fails: rebuild + `pip install -e lib/procgen` from the heist clone
    inside `ood` (does NOT affect `ood-stable`).
- [ ] `ssh rnn "ls /nas/ucb/czempin/data/goal-misgen/policy/icml/maze_afh/icml2_maze_exp0_0p/"` — sanity check existing maze layout.

### Step 3 — Smoke training run on rnn (4M timesteps)
- [ ] Submit:
  ```bash
  ssh rnn "cd /home/pavel/code/goal-misgen/heist-yrc-bench-fork && scripts/train_policies.sh -e heist -x 0 --random-percent 0 --num-timesteps 4000000 --days 1"
  ```
- [ ] SLURM job ID: _____
- [ ] Checkpoint appears at
      `/nas/ucb/czempin/data/goal-misgen/policy/icml/heist_afh/icml2_heist_exp0_0p/<timestamp>__seed_1111/model_*.pth`
- [ ] Wandb run shows non-trivial reward curve.
- [ ] `python -c "from scripts.common import get_checkpoints; print(get_checkpoints('heist', 0, '/nas/ucb/czempin/data/goal-misgen/policy/icml'))"`
  resolves the new path.

### Step 4 — Full 200M training (weak + strong)
- [ ] Submit weak:
  ```bash
  ssh rnn "cd /home/pavel/code/goal-misgen/heist-yrc-bench-fork && scripts/train_policies.sh -e heist -x 0 --random-percent 0"
  ```
  SLURM job ID: _____
- [ ] Submit strong:
  ```bash
  ssh rnn "cd /home/pavel/code/goal-misgen/heist-yrc-bench-fork && scripts/train_policies.sh -e heist -x 0 --random-percent 50"
  ```
  SLURM job ID: _____

### Step 5 — Post-training verification
- [ ] `get_checkpoints("heist", 0, "/nas/ucb/czempin/data/goal-misgen/policy/icml")`
      returns valid `weak` and `strong` paths (no `NOT_FOUND` markers).
- [ ] `find_best_model_checkpoint` finds `model_200015872.pth` in each
      (matches `EXPECTED_TIMESTEPS` in `scripts/common.py:6`).
- [ ] Rollout sanity check: weak agent gets high return on
      `heist_aisc_many_chests` and degraded return on `heist_aisc_many_keys`;
      strong agent gets reasonable return on both.

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

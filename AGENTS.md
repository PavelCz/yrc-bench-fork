# AGENTS.md / CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of YRC-Bench (Yield and Request Control Benchmark) focused exclusively on **Procgen environments**. The framework is for learning coordination strategies between novice (weak) and expert (strong) policies in reinforcement learning. The core problem is determining when to yield control from a weak agent to a strong agent, optimizing the trade-off between performance and query cost.

**Note:** While the codebase contains code for MiniGrid and Cliport environments from the original benchmark, this fork only uses Procgen. Focus on `YRC/envs/procgen/`, `configs/procgen_*.yaml`, and `YRC/checkpoints/procgen/` when working with this codebase.

### The paper this fork supports

This fork is the codebase for **"Getting by Goal Misgeneralization with a Little Help From an Expert"** (TMLR submission; project page https://asking-for-help-gmg.github.io/). The paper studies whether **goal misgeneralization (GMG)** in Procgen agents can be mitigated by letting a weak *novice* policy **ask a strong *expert* for help**. A coordination policy decides when to defer, and results are reported as return-vs-**AFHP** (ask-for-help percentage) curves.

Facts that matter when working in this repo:
- **Environments:** the paper's experiments use **`coinrun` and `maze`** (`maze_afh` at eval). `scripts/common.py:ENVS` is `["maze", "coinrun"]`. Checkpoints/configs for `heist` and other Procgen games exist in the tree but are not part of the paper's results.
- **Metric:** the paper uses **`level_afhp`** (fraction of episodes with any help). `step_afhp` (per-timestep) is an older definition retained for backward compatibility, not the paper's metric.
- **Two result families:** the main/recoverable benchmarks (§5.1) and the irrecoverable **"proxy-fail"** variants (§5.5, Procgen envs `coinrun_proxy_fail` / `maze_proxy_fail`), the latter being the paper's central limitation finding. Proxy-fail reuses the same trained policies — it is an eval-time environment change handled by `run_eval.py`, not separate training.
- **Maze expert:** maze main-curve results use a **robust expert trained from randomized agent-start positions** (`robust400`, 400M steps); see `scripts/common.py:get_robust_maze_strong_checkpoint` and `run_eval.py --robust400`. The non-robust maze expert appears only in the §5.3 brittleness analysis.
- **Seeds:** results use 4 independent seeds (experiment ids 0–3), reported as average + interquartile range.

## Reflective Memories

- Remember, that when you implement some algorithm I suggested or you came up with, that you will be biased towards thinking this alg is efficient or well-suited. However, this might not be true and it is hard to know how much you have thought about this. When writing conclusions or documentation, refrain from using language that subjectively assigns value to parts of the code or algorithms concept. Only use these, if we have some proof that these types of judements are correct.

## Development Environment

There are two conda environments:
- **`ood`**: For development and testing
- **`ood-stable`**: For running final experiments (frozen dependencies)

```bash
# Development (using conda run, recommended)
conda run -n ood <command>

# Or activate the environment first
conda activate ood

# Final experiments
conda run -n ood-stable <command>
```

## Key Commands

### Code Quality
```bash
# Format code
conda run -n ood ruff format <files>

# Lint code
conda run -n ood ruff check <files>

# Type check
conda run -n ood pytype <files>

# Run all checks on specific files
conda run -n ood ci/format_and_check.sh
```

### Training
```bash
# Train a coordination policy (Procgen)
python train_svdd.py -c configs/procgen_ood.yaml -n RUN_NAME -en ENV_NAME \
    -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
    -query_cost COST -cp_feature FEATURE_TYPE
```

The acting-policy training orchestrators (`scripts/train_policies.sh`, `scripts/train_ensemble_policies.sh`) submit Procgen PPO jobs via SLURM. On the RNN cluster these runs under-use a whole GPU, so both scripts accept `--gpu-shards N` (fractional GPU), `--cpus-per-task N`, and `--mem SIZE` to pack several runs per card — see [docs/gpu_sharding.md](docs/gpu_sharding.md) for the shard topology, measured resource footprint, and recommended settings.

### Evaluation
```bash
# Evaluate AFHP / return curves for a coordination policy
python eval_afhp.py -c configs/eval/coinrun/max_prob.yaml -n RUN_NAME \
    -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
    -level_seeds_file PATH/TO/level_seeds.json -coverage_fraction 0.05

# Evaluate a standalone acting policy checkpoint directly
python eval_policy.py -c configs/procgen_threshold.yaml \
    --model_file PATH/TO/MODEL.pth -num_rollouts 100

# Submit a batch of AFHP evaluations via SLURM
python scripts/run_eval.py --env coinrun --method max-prob --exp-ids 0 1 2 3
```

### Smoke tests on rnn

End-to-end smoke tests against the frozen `ood-stable` env run on the
`rnn` server via SSH + `scripts/run_eval.py`. See
[`docs/rnn_smoke_tests.md`](docs/rnn_smoke_tests.md) for the full
workflow: prerequisites, pulling the latest commit, submitting a job,
log paths under `/nas/ucb/czempin/data/goal-misgen/slurm-logs/`, tailing
in real time, cancelling, and baseline timing. The same SSH/conda
wrapper is used for one-off diagnostic scripts on rnn.

### Analysis
```bash
# Generate plots
python -m analyzing.paper_plot --eval_dir PATH/TO/EVALS --env coinrun
```

### Installation
```bash
# Install main requirements
pip install -r requirements.txt

# Install environment libraries (e.g., Procgen)
pip install -e lib/LIBRARY_NAME
```

## Architecture Overview

### Core Components

1. **YRC/algorithms/**: Coordination algorithms
   - `ood.py`: Out-of-distribution detection methods (Deep SVDD)
   - `rl.py`: Reinforcement learning-based coordination (PPO)
   - `threshold.py`: Threshold-based methods using confidence metrics
   - `random.py`: Baseline strategies (random, always weak/strong)

2. **YRC/core/**: Essential infrastructure
   - `evaluator.py`: Unified evaluation framework with adaptive threshold sampling
   - `configs/config.py`: Configuration data structure
   - `configs/utils.py`: Configuration loading and command-line override handling
   - `rollout_helper.py`: Utility for collecting rollouts
   - `algorithm.py`: Base algorithm interface
   - `environment.py`: Environment factory and wrappers
   - `policy.py`: Policy factory and interfaces

3. **YRC/coverage/**: AFHP coverage samplers
   - `coverage_search.py`: Binary-search-based threshold sampling for `step_afhp` and `level_afhp`

4. **YRC/envs/**: Environment wrappers
   - `procgen/`: Procgen environments with models, policies, and wrappers (primary focus of this fork)

5. **YRC/policies/**: Policy implementations
   - `base.py`: `TimestepRandomPolicy`, `LevelBasedRandomPolicy`, `AlwaysPolicy`
   - `heuristic.py`: `ExponentialHeuristicPolicy`, `WaitPolicy`
   - `threshold.py`: `ThresholdPolicy` (confidence-based: `max_prob`, `max_logit`, `ensemble_variance`)
   - `ood.py`: `OODPolicy` (Deep SVDD, AutoEncoder)
   - `rl.py`: RL-based coordination policies

### Configuration System

The project uses hierarchical YAML configs in `configs/`:
- `common.yaml`: Common parameters shared across experiments
- `procgen_*.yaml`: Procgen-specific configs (primary configs for this fork)
- Algorithm-specific parameters embedded in each config
- Checkpoint paths for pre-trained acting policies go in `YRC/checkpoints/procgen/`

### Key Design Patterns

1. **Factory Pattern**: Environment and policy factories for modular instantiation
2. **Configuration-Driven**: All experiments driven by YAML configs with command-line overrides
3. **Unified Evaluation**: Single evaluator handles all coordination methods
4. **Adaptive Sampling**: Sophisticated threshold evaluation algorithm

### Important Implementation Details

1. **Threshold Evaluation**: The evaluator uses a 2D adaptive sampling algorithm that ensures good coverage across both AFHP (ask-for-help percentage) and return axes:
   - Phase 1: Evaluates boundaries (always/never ask for help)
   - Phase 2: Samples critical percentiles (5, 10, 25, 50, 75, 90, 95)
   - Phase 3: Uses Delaunay triangulation to identify and fill coverage gaps
   - Phase 4: Refines return axis to ensure smooth curves
   - Maintains backward compatibility with legacy format

2. **AFHP Metrics and Percentile Calibration**: Two AFHP metrics exist: **step_afhp** (fraction of timesteps) and **level_afhp** (fraction of episodes with any help). All policies implement `train_percentile_step(p)` and `train_percentile_level(p)` to map percentiles to thresholds calibrated for each metric. Calibration runs in `calibrate_percentile_mapping()` in `eval_afhp.py` before the sampler starts. See `docs/percentile_calibration.md` for the full support matrix, per-policy formulas, and calibration data sources.

3. **Feature Types**: Coordination policies can use:
   - Raw observations (`obs`)
   - Weak agent's hidden features (`hidden`)
   - Weak agent's action distributions / logits (`dist`)
   - Combinations (`hidden_obs`, `hidden_dist`, `obs_dist`, `obs_hidden_dist`)

4. **Memory Management**: Recent work focuses on efficient handling of large rollout datasets, especially for OOD detection methods that require storing and processing many samples.

5. **Experiment Tracking**: All experiments are tracked with Weights & Biases (wandb) for reproducibility. The tracking includes:
   - Training metrics and curves
   - Evaluation videos with score bars
   - Hyperparameters and configurations
   - Model checkpoints

6. **Checkpoint Management**: Three types of checkpoints are saved during training:
   - `best_val_sim.ckpt`: Best validation performance on simulated weak agent
   - `best_val_true.ckpt`: Best validation performance on true weak agent
   - `last.ckpt`: Most recent checkpoint

7. **Acting Policy Requirements**: Pre-trained acting policies (sim weak, weak, strong) must be provided for most environments. These should be placed in `YRC/checkpoints/{environment}/` following the existing structure.

8. **Procgen Evaluation Flow**: AFHP evaluation in this fork goes through `eval_afhp.py`, which calibrates percentile-to-threshold mappings, then calls `YRC/coverage/coverage_search.py` to sample thresholds adaptively. Batch evaluation is typically launched via `scripts/run_eval.py`. See `docs/adaptive_coverage_sampling.md` and `docs/percentile_calibration.md` for the current behavior.

## Documentation

Reference docs live in `docs/`:
- [gpu_sharding.md](docs/gpu_sharding.md) — fractional-GPU (`shard`) training on RNN: cluster topology, measured VRAM/CPU/RAM footprint, and recommended `--gpu-shards`/`--cpus-per-task`/`--mem` settings.
- [rnn_smoke_tests.md](docs/rnn_smoke_tests.md) — end-to-end smoke-test workflow on the `rnn` server (SSH + `scripts/run_eval.py`, log paths, tailing/cancelling).
- [container_workflow.md](docs/container_workflow.md) — Docker→Apptainer workflow used on CARC.
- [level_seed_splits.md](docs/level_seed_splits.md) — how `policy_train`, `ood_train`, `validation`, and `ood_eval` seed splits flow through the pipeline.
- [percentile_calibration.md](docs/percentile_calibration.md) — percentile→threshold calibration (support matrix, per-policy formulas, data sources).
- [adaptive_coverage_sampling.md](docs/adaptive_coverage_sampling.md) — the AFHP coverage sampler behavior.
- [bisection_boundary_bug.md](docs/bisection_boundary_bug.md), [image_svdd_collapse_bugs.md](docs/image_svdd_collapse_bugs.md) — SVDD bug write-ups.
- [known_issues.md](docs/known_issues.md) — open issues.

## GitHub issues (no `gh` CLI)

Project tracking (e.g. the additional-seeds campaign) lives in GitHub issues on **`PavelCz/yrc-bench-fork`**. The `gh` CLI is **not installed** on the dev box, but a GitHub OAuth token is available in `~/.config/gh/hosts.yml`, and the git remote authenticates over SSH as `PavelCz`. Use the REST API directly:

```bash
# Grab the token (never print or commit it)
TOKEN=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.path.expanduser('~/.config/gh/hosts.yml')))['github.com']['oauth_token'])")
REPO=PavelCz/yrc-bench-fork
AUTH=(-H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json")

# List open issues (number + title)
curl -sS "${AUTH[@]}" "https://api.github.com/repos/$REPO/issues?state=open" \
  | python3 -c "import json,sys; [print(i['number'], i['title']) for i in json.load(sys.stdin) if 'pull_request' not in i]"

# Read one issue (body + metadata)
curl -sS "${AUTH[@]}" "https://api.github.com/repos/$REPO/issues/20"

# Comment on an issue
curl -sS "${AUTH[@]}" -X POST "https://api.github.com/repos/$REPO/issues/20/comments" \
  -d "$(python3 -c 'import json;print(json.dumps({"body":"..."}))')"

# Create an issue / edit a body: POST .../issues  /  PATCH .../issues/N
```

Creating issues and posting comments is outward-facing — do it only when the user asks.

## Python Best Practices

- Use Pathlib to interact with paths instead of the os package
- Follow the code style enforced by ruff (format and check)
- Add type hints where possible and run pytype for verification
- Use the existing factory patterns when adding new components

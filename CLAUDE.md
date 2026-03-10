# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of YRC-Bench (Yield and Request Control Benchmark) focused exclusively on **Procgen environments**. The framework is for learning coordination strategies between novice (weak) and expert (strong) policies in reinforcement learning. The core problem is determining when to yield control from a weak agent to a strong agent, optimizing the trade-off between performance and query cost.

**Note:** While the codebase contains code for MiniGrid and Cliport environments from the original benchmark, this fork only uses Procgen. Focus on `YRC/envs/procgen/`, `configs/procgen_*.yaml`, and `YRC/checkpoints/procgen/` when working with this codebase.

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
python train.py -c configs/procgen_ood.yaml -n RUN_NAME -en ENV_NAME \
    -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
    -query_cost COST -cp_feature FEATURE_TYPE
```

### Evaluation
```bash
# Evaluate a trained model
python eval.py -c configs/CONFIG.yaml -n RUN_NAME -en ENV_NAME \
    -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
    -query_cost COST -f_n CHECKPOINT_NAME -seed SEED

# Evaluate thresholds systematically
python eval_thresholds.py --config CONFIG.yaml --eval.threshold_bins 20
```

### Analysis
```bash
# Parse raw results
python analyzing/parse.py

# Aggregate results
python analyzing/aggregate.py

# Generate plots
python analyzing/fig*.py
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
   - `ood.py`: Out-of-distribution detection methods (Deep SVDD, Mahalanobis AE)
   - `lightning_ae.py`: Lightning-based autoencoder for OOD detection
   - `rl.py`: Reinforcement learning-based coordination (PPO)
   - `threshold.py`: Threshold-based methods using confidence metrics
   - `random.py`: Baseline strategies (random, always weak/strong)

2. **YRC/core/**: Essential infrastructure
   - `evaluator.py`: Unified evaluation framework with adaptive threshold sampling
   - `config.py`: Configuration management system
   - `dataset.py`: Rollout data handling
   - `rollout_helper.py`: Utility for collecting rollouts
   - `algorithm.py`: Base algorithm interface
   - `environment.py`: Environment factory and wrappers
   - `policy.py`: Policy factory and interfaces

3. **YRC/envs/**: Environment wrappers
   - `procgen/`: Procgen environments with models, policies, and wrappers (primary focus of this fork)

4. **YRC/policies/**: Policy implementations
   - `base.py`: `TimestepRandomPolicy`, `LevelBasedRandomPolicy`, `AlwaysPolicy`
   - `heuristic.py`: `ExponentialHeuristicPolicy`, `WaitPolicy`
   - `threshold.py`: `ThresholdPolicy` (confidence-based: `max_prob`, `max_logit`, `ensemble_variance`)
   - `ood.py`: `OODPolicy` (Deep SVDD, AutoEncoder)
   - `lightning_ae.py`: `LightningAEPolicy` (PyTorch Lightning autoencoders)
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

2. **AFHP Metrics and Percentile Calibration**: There are two AFHP (ask-for-help percentage) metrics:
   - **step_afhp**: fraction of *timesteps* where help was requested
   - **level_afhp**: fraction of *episodes* where help was requested at least once

   All policies implement two calibration methods instead of the old `train_percentile()`:
   - `train_percentile_step(p)` — returns a threshold calibrated for step_afhp
   - `train_percentile_level(p)` — returns a threshold calibrated for level_afhp

   Not all policies support both. Unsupported variants raise `NotImplementedError`:

   | Policy | `_step` | `_level` |
   |---|---|---|
   | `ThresholdPolicy` | per-step scores | per-episode max scores |
   | `TimestepRandomPolicy` | linear | `1 - p^(1/L)` formula |
   | `LevelBasedRandomPolicy` | NotImplementedError | linear |
   | `ExponentialHeuristicPolicy` | NotImplementedError | linear |
   | `WaitPolicy` | timestep threshold | NotImplementedError |
   | `OODPolicy` | training decision scores | NotImplementedError |
   | `LightningAEPolicy` | training decision scores | NotImplementedError |

   `ThresholdPolicy.generate_scores()` collects both per-step scores (`_train_scores`) and per-episode max scores (`_train_episode_max_scores`). The episode-max approach means that `np.percentile(episode_max_scores, 90)` directly gives the threshold where 10% of episodes have any step exceeding it.

   The samplers in `YRC/coverage/coverage_search.py` call these methods directly:
   - `create_level_afhp_threshold_sampler` → `train_percentile_level`
   - `create_step_afhp_threshold_sampler` → `train_percentile_step`

4. **Feature Types**: Coordination policies can use:
   - Raw observations (`obs`)
   - Weak agent's hidden features (`feature`)
   - Weak agent's action distributions (`action`)
   - Combinations (e.g., `obs+feature`, `obs+action`, `feature+action`, `obs+feature+action`)

5. **Memory Management**: Recent work focuses on efficient handling of large rollout datasets, especially for OOD detection methods that require storing and processing many samples.

6. **Experiment Tracking**: All experiments are tracked with Weights & Biases (wandb) for reproducibility. The tracking includes:
   - Training metrics and curves
   - Evaluation videos with score bars
   - Hyperparameters and configurations
   - Model checkpoints

7. **Checkpoint Management**: Three types of checkpoints are saved during training:
   - `best_val_sim.ckpt`: Best validation performance on simulated weak agent
   - `best_val_true.ckpt`: Best validation performance on true weak agent
   - `last.ckpt`: Most recent checkpoint

8. **Acting Policy Requirements**: Pre-trained acting policies (sim weak, weak, strong) must be provided for most environments. These should be placed in `YRC/checkpoints/{environment}/` following the existing structure.

## Python Best Practices

- Use Pathlib to interact with paths instead of the os package
- Follow the code style enforced by ruff (format and check)
- Add type hints where possible and run pytype for verification
- Use the existing factory patterns when adding new components
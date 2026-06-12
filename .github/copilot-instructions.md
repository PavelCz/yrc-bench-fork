# YRC-Bench (Yield and Request Control Benchmark)

YRC-Bench is a research framework for learning coordination strategies between novice (weak) and expert (strong) policies in reinforcement learning environments. The core problem is determining when to yield control from a weak agent to a strong agent, optimizing the trade-off between performance and query cost.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Repository Setup and Dependencies
- Clone the repository with submodules:
  ```bash
  git submodule update --init --recursive
  ```
  This step is CRITICAL - takes ~2-3 minutes. NEVER CANCEL. The repo requires custom environment libraries.

- Install core dependencies:
  ```bash
  pip install wandb jsonargparse PyYAML torch numpy scipy scikit-learn matplotlib pillow tqdm pytorch-lightning gymnasium ruff pytype
  ```
  Note: Dependency installation may fail due to network timeouts, particularly for packages like `numba`, `llvmlite`, or packages from the main requirements.txt. This is a known limitation.

- Install environment-specific requirements if needed:
  ```bash
  pip install -r requirements_minigrid.txt  # For MiniGrid environments only
  ```
  WARNING: This may fail due to network timeouts. Document any failures clearly.

### Code Quality and Validation
- Format and lint code (checks specific files only):
  ```bash
  ./ci/format_and_check.sh
  ```
  Takes ~6 seconds on subsequent runs (first run may take longer). NEVER CANCEL. Uses ruff for formatting/linting and pytype for type checking.
  
  **NOTE**: This script only checks a subset of files:
  - YRC/core/rollout_helper.py
  - eval_thresholds.py

- Check entire codebase (will show many linting issues):
  ```bash
  ruff check YRC/              # Check all YRC files - shows 42+ errors
  ruff format --check YRC/     # Check formatting - 27+ files need reformatting
  ```

- Individual commands:
  ```bash
  ruff format <files>    # Format specific files
  ruff check <files>     # Lint specific files  
  pytype <files>         # Type check (may take longer)
  ```

### Analysis and Results Processing
- Parse raw results:
  ```bash
  python analyzing/parse.py
  ```

- Aggregate results:
  ```bash
  python analyzing/aggregate.py
  ```

- Generate plots (various figure generation scripts):
  ```bash
  python analyzing/fig*.py
  ```

- Evaluate thresholds systematically (requires full dependencies):
  ```bash
  python eval_thresholds.py --config CONFIG.yaml --eval.threshold_bins 20
  ```
  Note: This requires numba and other ML dependencies that may fail to install.
### Training and Evaluation
  ```bash
  python train.py -c configs/CONFIG.yaml -n RUN_NAME -en ENV_NAME \
      -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
      -query_cost COST -cp_feature FEATURE_TYPE
  ```

- Evaluate a trained model:
  ```bash
  python eval.py -c configs/CONFIG.yaml -n RUN_NAME -en ENV_NAME \
      -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt \
      -query_cost COST -f_n CHECKPOINT_NAME -seed SEED
  ```

- For Cliport environments (oracle-based strong agent):
  ```bash
  python train.py -c configs/cliport_ood.yaml -n RUN_NAME -en ENV_NAME \
      -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt \
      -cp_feature FEATURE_TYPE -query_cost COST
  ```

### Validation Workflows
- ALWAYS run code quality checks before committing:
  ```bash
  ./ci/format_and_check.sh
  ```

- Test basic imports after setup:
  ```bash
  python -c "import YRC.core.configs; print('Core imports OK')"
  ```

- Validate training script help:
  ```bash
  python train.py -h
  ```
  Note: This works with basic dependencies.

- Validate evaluation script help (may fail without numba):
  ```bash
  python eval.py -h
  ```
  Note: This requires additional dependencies like numba that may fail to install due to network issues.

## Key Architecture Components

### Core Directories
- `YRC/algorithms/`: Coordination algorithms (OOD detection, RL-based, threshold methods)
- `YRC/core/`: Essential infrastructure (evaluator, config management, environment wrappers)
- `YRC/envs/`: Environment-specific wrappers for MiniGrid, Procgen, Cliport
- `YRC/policies/`: Policy implementations for all coordination methods
- `YRC/models/`: Neural network models and utilities
- `configs/`: YAML configuration files for all experiments
- `lib/`: Git submodules for custom environment libraries
- `analyzing/`: Results analysis and plotting scripts
- `ci/`: Code quality and formatting tools

### Configuration System
- All experiments driven by YAML configs in `configs/`
- `common.yaml` contains shared parameters
- Environment-specific configs: `minigrid_*.yaml`, `procgen_*.yaml`, `cliport_*.yaml`
- Command-line overrides available for all parameters
- Checkpoint paths for pre-trained acting policies go in `YRC/checkpoints/`

### Supported Components
- **Environment Suites**: MiniGrid, Procgen, Cliport
- **Algorithms**: Random baselines, threshold-based, OOD detection (DeepSVDD), RL-based (PPO)
- **Feature Types**: Raw observations (`obs`), hidden features (`feature`), action distributions (`action`), combinations

## Common Issues and Limitations

### Installation Challenges
- Network timeouts are common when installing ML dependencies
- The requirements.txt has version conflicts with old packages (e.g., absl-py==0.7.0 requires Python 2.7+)
- Some custom libraries in `lib/` may not install properly due to network issues
- Document any installation failures clearly in your changes

### Build and Test Information
- No comprehensive test suite exists
- Primary validation is through code quality tools (ruff, pytype)
- Training/evaluation require pre-trained acting policies (not included in repo)
- Actual model training may take hours and requires GPU resources

### Working Without Full Dependencies
- Core config and utility modules can be imported with basic dependencies
- Code quality checks work with ruff and pytype
- Training/evaluation scripts require full ML stack and may not work in limited environments

## Important Validation Steps
1. Always run `git submodule update --init --recursive` after cloning
2. Test basic imports: `python -c "import YRC.core.configs"`
3. Run code quality checks: `./ci/format_and_check.sh`
4. Validate training script help: `python train.py -h` (works with basic dependencies)
5. Validate evaluation script help: `python eval.py -h` (requires full dependencies)
6. Document any dependency installation failures clearly
7. Never cancel long-running operations (submodule init, dependency installs)

## Common Commands Reference

### Configuration Examples
- Random baseline: `configs/minigrid_random.yaml`
- Threshold-based: `configs/procgen_threshold.yaml`
- OOD detection: `configs/cliport_ood.yaml`
- RL-based: `configs/procgen_skyline.yaml`

### Feature Types Available
- `obs`: Raw observations
- `hidden`: Hidden features from weak policy
- `dist`: Action distributions from weak policy
- `hidden_obs`: Combined hidden features + observations
- `obs_dist`: Combined observations + action distributions
- `hidden_dist`: Combined hidden features + action distributions
- `obs_hidden_dist`: All three combined

### Supported Algorithms
- `random`: Random coordination decisions
- `threshold`: Confidence-based thresholds (max_logit, max_prob, margin, neg_entropy, neg_energy)
- `ood`: Out-of-distribution detection (DeepSVDD, AutoEncoder)
- `rl`: Reinforcement learning-based coordination (PPO)

## Repository Context
- Research codebase for academic paper: https://arxiv.org/abs/2502.09583
- Modular and extensible design for adding new environments/algorithms
- Uses Weights & Biases (wandb) for experiment tracking
- Supports distributed training with PyTorch Lightning
- Three types of checkpoints saved: best_val_sim.ckpt, best_val_true.ckpt, last.ckpt
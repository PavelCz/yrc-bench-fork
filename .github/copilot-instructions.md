# YRC-Bench: Yielding and Requesting Control Benchmark

YRC-Bench is a Python machine learning benchmark for training coordination strategies between weak and strong reinforcement learning policies across multiple environments (MiniGrid, Procgen, Cliport).

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Critical Installation Issues

**NETWORK TIMEOUT ISSUE**: pip installations from PyPI frequently timeout in this environment. This is a known limitation that affects dependency installation. All timeout values below account for this issue.

## Working Effectively

### Bootstrap the Repository
- `git submodule update --init --recursive` -- takes 7 seconds. Initializes 4 submodules (Minigrid, cliport, procgenAISC, pyod).

### Environment Setup
- Requires Python 3.8+ (tested with Python 3.12.3)
- Create virtual environment: `python3 -m venv venv && source venv/bin/activate`
- **DEPENDENCY INSTALLATION FREQUENTLY FAILS**: pip install commands timeout due to network issues. NEVER CANCEL these commands.

### Install Dependencies (Known Issues)
- `pip install -r requirements.txt` -- **FAILS due to network timeouts with PyPI**. Set timeout to 60+ minutes. NEVER CANCEL.
- `pip install -r requirements_minigrid.txt` -- **FAILS due to network timeouts**. Set timeout to 60+ minutes. NEVER CANCEL.
- Local library installation: `pip install -e lib/LIBRARY_NAME` -- **Also fails due to network timeouts**.

### Alternative Installation Approach
When pip fails with network timeouts (common in this environment):
1. Try installing individual packages with shorter timeouts: `pip install package_name --timeout 30`
2. Use system Python packages when available
3. Install from local directories when possible
4. Document failed installations as known issues

### Key Dependencies (when installation works)
Core packages needed: jsonargparse, PyYAML, torch, numpy, pandas, matplotlib, gymnasium

### Main Commands (once dependencies are installed)
- Training: `python train.py -c configs/CONFIG.yaml -n RUN_NAME -en ENV_NAME -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt -query_cost COST -cp_feature FEATURE_TYPE`
- Evaluation: `python eval.py -c configs/CONFIG.yaml -n RUN_NAME -en ENV_NAME -sim PATH/TO/SIM_WEAK.pt -weak PATH/TO/WEAK.pt -strong PATH/TO/STRONG.pt -query_cost COST -cp_feature FEATURE_TYPE -f_n CHECKPOINT_NAME -seed SEED`
- Help: `python train.py -h` and `python eval.py -h` (requires dependencies)

### Expected Build/Install Times
- Git submodule initialization: 7 seconds
- Virtual environment creation: <5 seconds  
- Dependency installation (when working): 30-60 minutes. NEVER CANCEL. Set timeout to 90+ minutes.
- **CRITICAL**: Network timeouts are common and cause most installations to fail

### Checkpoints Required
The benchmark requires pre-trained model checkpoints in `YRC/checkpoints/` directory (not included in repository). Download from [Google Drive Link](https://drive.google.com/file/d/1Tix3PO8gwJljwHLcolaQo5Moaci-apF8/view?usp=sharing).

## Validation and Testing

### Manual Validation Requirements
- **Cannot fully validate without dependencies**: Network issues prevent complete installation
- Basic structure validation: file exploration, config file examination
- **No functional testing possible** until dependency installation succeeds

### Testing Scenarios (when dependencies available)
- Test basic help commands: `python train.py -h`, `python eval.py -h`
- Run simple random algorithm: Use configs like `configs/minigrid_random.yaml`
- Validate with minimal environments before full training runs

### Known Limitations
- **pip installations fail due to network timeouts** - this is documented and expected
- Full functional testing cannot be performed without working dependencies
- Model checkpoints must be downloaded separately 

## Directory Structure and Navigation

### Repository Root
```
├── README.md           # Main documentation
├── train.py           # Main training script (26 lines)
├── eval.py            # Evaluation script (20 lines)  
├── flags.py           # Command line argument definitions (85 lines)
├── requirements.txt   # Main dependencies (fails to install)
├── requirements_minigrid.txt # MiniGrid-specific deps (fails to install)
├── configs/           # 22 YAML configuration files
├── YRC/               # Core benchmark code
├── lib/               # Git submodules (Minigrid, cliport, procgenAISC, pyod)
├── analyzing/         # Analysis and plotting scripts (10 files)
└── experiments/       # Output directory (initially empty)
```

### Key Configuration Files
- `configs/common.yaml` - Shared configuration settings
- `configs/{suite}_{algorithm}.yaml` - Environment/algorithm combinations
  - Suites: minigrid, procgen, cliport
  - Algorithms: always, random, threshold, skyline, ood

### Core YRC Module
- `YRC/core/` - Main framework (algorithm.py, environment.py, policy.py, evaluator.py)
- `YRC/algorithms/` - Algorithm implementations  
- `YRC/policies/` - Policy implementations
- `YRC/envs/` - Environment wrappers
- `YRC/models/` - Neural network models

### Supported Environments
- **MiniGrid**: DistShift, DoorKey, LavaGap
- **Procgen**: bossfight, caveflyer, chaser, climber, coinrun, dodgeball, heist, jumper, maze, ninja, plunder  
- **Cliport**: assembling-kits-seq, packing-boxes-pairs, put-block-in-bowl, stack-block-pyramid-seq, separating-piles

### Supported Algorithms
- **Baselines**: Random, Always (weak/strong)
- **Threshold-based**: max_logit, max_prob, margin, neg_entropy, neg_energy
- **OOD-detection**: DeepSVDD with various feature types
- **RL-based**: PPO with various feature combinations

### Feature Types (for OOD and RL methods)
- `obs` - Raw image observations
- `hidden` - Weak agent hidden features  
- `dist` - Weak agent action distributions
- `hidden_obs`, `obs_dist`, `hidden_dist`, `obs_hidden_dist` - Combinations

## Common Tasks

### Example Training Commands (when dependencies work)
```bash
# Random algorithm on MiniGrid DoorKey
python train.py -c configs/minigrid_random.yaml -n DoorKey_random_qc02 -en MiniGrid-DoorKey -sim YRC/checkpoints/minigrid/DoorKey/sim_weak/status.pt -weak YRC/checkpoints/minigrid/DoorKey/weak/status.pt -strong YRC/checkpoints/minigrid/DoorKey/strong/status.pt -query_cost 0.2 -en_tr_suffix=-5x5-v0 -en_te_suffix=-8x8-v0

# Threshold algorithm on Procgen coinrun  
python train.py -c configs/procgen_threshold.yaml -n coinrun_threshold_margin_qc06 -en coinrun -sim YRC/checkpoints/procgen/coinrun/sim_weak/model_40009728.pth -weak YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth -strong YRC/checkpoints/procgen/coinrun/strong/model_200015872.pth -cp_metric margin -query_cost 0.6
```

### Analysis Workflow (when experiments complete)
1. Run experiments and save to `experiments/` directory
2. Edit `analyzing/constants.py` with your environment/algorithm settings  
3. Extract results: `python analyzing/parse.py`
4. Aggregate results: `python analyzing/aggregate.py`  
5. Generate plots: Run scripts in `analyzing/` (fig3_wins.py, fig4_ratio_to_rl.py, etc.)

### Git Submodule Management
- Check status: `git submodule status`
- Update: `git submodule update --remote`  
- Individual update: `git submodule update lib/LIBRARY_NAME`

## Troubleshooting

### Network Timeout Issues
- **Expected behavior**: pip installations timeout frequently
- **Solution**: Document as known limitation, use alternative installation methods
- **DO NOT**: Keep retrying failed pip commands without addressing root cause

### Missing Checkpoints
- Download from Google Drive link in README
- Extract to `YRC/checkpoints/` directory
- Verify file paths match config files

### Import Errors
- Usually due to failed dependency installation
- Check virtual environment activation
- Verify Python version compatibility (3.8+)

### Configuration Issues
- Validate YAML syntax in config files
- Check file paths in configurations match actual locations
- Ensure environment names match available environments

## Development Notes

- The benchmark uses YAML configuration files extensively - always validate syntax
- Model checkpoints are large and stored separately from code
- Experiments generate significant data in the `experiments/` directory
- Analysis scripts expect specific directory structures and naming conventions
- Network limitations significantly impact development workflow in this environment
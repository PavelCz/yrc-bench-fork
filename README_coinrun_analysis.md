# Coinrun Counterfactual Analysis

This directory contains the implementation for coinrun counterfactual analysis as requested in the issue.

## Overview

The coinrun counterfactual analysis performs the following steps:
1. Roll out the weak agent in the coinrun environment with randomly placed coin (`random_percent=100`)
2. Iteratively test different level seeds until finding one where the weak agent fails
3. Run the same level seed with deterministic coin placement (`random_percent=0`)
4. Save both rollouts as videos for analysis
5. Save the level seed and results for reference

## Files

- `coinrun_counterfactual_analysis.py` - Main analysis script
- `test_dependencies.py` - Dependency checker script
- `README.md` - This documentation

## Requirements

### Dependencies
The following Python packages are required:
- `numpy` - For numerical operations
- `torch` - For loading and running the weak agent model
- `opencv-python` - For video recording
- `gym3` - Required by procgen environment
- YRC framework components (included in this repository)
- ProcgenAISC environment (submodule in `lib/procgenAISC`)

### Installation
1. Install basic Python dependencies:
   ```bash
   pip install numpy torch opencv-python gym3
   ```

2. Install the procgen environment:
   ```bash
   pip install -e lib/procgenAISC
   ```

3. Ensure the weak agent checkpoint is available at:
   ```
   YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth
   ```

### Checkpoint Setup
According to the YRC-Bench documentation, checkpoints should be downloaded from:
[Google Drive Link](https://drive.google.com/file/d/1Tix3PO8gwJljwHLcolaQo5Moaci-apF8/view?usp=sharing)

Extract the checkpoints to the `YRC/checkpoints/` directory.

## Usage

### Check Dependencies
First, verify all dependencies are installed:
```bash
python test_dependencies.py
```

### Run Analysis
Execute the counterfactual analysis:
```bash
python coinrun_counterfactual_analysis.py --output_dir my_analysis_results
```

### Command Line Options
- `--weak_agent_path` - Path to weak agent checkpoint (default: YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth)
- `--output_dir` - Directory to save results and videos (default: coinrun_analysis_output)
- `--device` - Device to run on: cpu or cuda (default: cpu)
- `--max_seed_attempts` - Maximum seeds to try when finding failure case (default: 100)
- `--start_seed` - Starting seed value for search (default: 0)

### Example
```bash
# Run analysis with custom parameters
python coinrun_counterfactual_analysis.py \
  --output_dir /path/to/results \
  --device cuda \
  --max_seed_attempts 50 \
  --start_seed 1000
```

## Output

The script creates the following outputs in the specified directory:

1. **Videos**: 
   - `seed_X_random_placement.mp4` - Rollout with random coin placement
   - `seed_X_deterministic_placement.mp4` - Rollout with deterministic coin placement

2. **Results**:
   - `analysis_results_seed_X.json` - Detailed analysis results including rewards, episode lengths, and success status

3. **Logs**:
   - `coinrun_analysis.log` - Detailed execution log

## Analysis Results Structure

The JSON results file contains:
```json
{
  "seed": 123,
  "timestamp": "2025-01-XX...",
  "random_placement": {
    "reward": 0.0,
    "episode_length": 150,
    "success": false,
    "num_frames": 150
  },
  "deterministic_placement": {
    "reward": 10.0,
    "episode_length": 180,
    "success": true,
    "num_frames": 180
  }
}
```

## Implementation Details

### Key Parameters
- `random_percent=100` - Fully randomized coin placement
- `random_percent=0` - Deterministic coin placement
- Level seeds control the environment layout while maintaining consistent agent behavior

### Environment Setup
The script uses the YRC framework's procgen environment wrapper with:
- Single environment (`num_envs=1`) for deterministic rollouts
- Hard difficulty mode for challenging scenarios
- Consistent level seeds for reproducible comparisons

### Video Recording
Videos are recorded as MP4 files at 30 FPS using OpenCV, capturing the RGB observations from the environment.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Run `test_dependencies.py` to identify missing packages
2. **Checkpoint Not Found**: Download and extract checkpoints as described above
3. **Environment Setup**: Ensure git submodules are initialized (`git submodule update --init --recursive`)
4. **Memory Issues**: Use `--device cpu` if CUDA is not available or causes issues

### Debugging
- Check the log file in the output directory for detailed execution information
- Verify environment creation by testing with a simple seed first
- Ensure the weak agent loads correctly by checking the checkpoint path

## Expected Behavior

The script should:
1. Successfully find a level seed where the weak agent fails with random coin placement
2. Show improved performance on the same seed with deterministic coin placement
3. Generate videos demonstrating the behavioral differences
4. Complete analysis within a reasonable time (usually < 10 minutes for default settings)

This counterfactual analysis helps understand how coin placement randomization affects agent performance and can reveal cases where the agent's failure is due to environmental stochasticity rather than fundamental policy limitations.
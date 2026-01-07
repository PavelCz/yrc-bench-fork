# RandomEnvSwitchWrapper Usage

This document explains how to use the `RandomEnvSwitchWrapper` feature in `train.py` to train agents on multiple Procgen environments with specified probabilities.

## Overview

The `RandomEnvSwitchWrapper` allows you to train a single agent on two different Procgen environments simultaneously. During training, each parallel environment randomly switches between the two specified environments based on a given probability.

## Command-Line Arguments

Two new arguments have been added to `train.py`:

- `--switch_env_names`: List of exactly 2 environment names to randomly switch between
- `--switch_percent`: Percentage (0-100) for choosing the first environment (default: 50)

## Usage Examples

### Example 1: Train on CoinRun and StarPilot (50/50 split)

```bash
python train.py \
    --exp_name coinrun_starpilot_50_50 \
    --switch_env_names coinrun starpilot \
    --switch_percent 50 \
    --param_name easy-200 \
    --num_timesteps 25000000 \
    --seed 42
```

### Example 2: Train on CoinRun and BigFish (70/30 split)

```bash
python train.py \
    --exp_name coinrun_bigfish_70_30 \
    --switch_env_names coinrun bigfish \
    --switch_percent 70 \
    --param_name easy-200 \
    --num_timesteps 25000000 \
    --seed 123
```

This will train with 70% probability of using CoinRun and 30% probability of using BigFish.

### Example 3: Train on Maze and Heist (40/60 split)

```bash
python train.py \
    --exp_name maze_heist_40_60 \
    --switch_env_names maze heist \
    --switch_percent 40 \
    --param_name hard \
    --num_timesteps 10000000
```

This will train with 40% probability of using Maze and 60% probability of using Heist.

## How It Works

1. When `--switch_env_names` is provided, `train.py` creates two separate `ProcgenEnv` instances
2. Both environments are configured with the same hyperparameters (num_levels, start_level, distribution_mode, etc.)
3. The two environments are wrapped with `RandomEnvSwitchWrapper`
4. During training:
   - On each episode reset, each parallel environment randomly selects which of the two environments to use
   - The selection is based on the specified `--switch_percent` probability
   - This allows the agent to experience both environments during training

## Notes

- The wrapper requires **exactly 2 environments**. For more complex multi-environment training, see `train-interleave-envs.py`
- The validation environment does NOT use the wrapper - it uses the standard `--val_env_name` or `--env_name`
- Both environments must be compatible (same action space) for the wrapper to work correctly
- The `--switch_percent` value represents the probability of choosing the **first** environment in the list

## Available Procgen Environments

The following Procgen environments can be used:
- bigfish
- bossfight
- caveflyer
- chaser
- climber
- coinrun
- dodgeball
- fruitbot
- heist
- jumper
- leaper
- maze
- miner
- ninja
- plunder
- starpilot

## Implementation Details

The implementation wraps procgen's official `RandomEnvSwitchWrapper` class, which:
- Maintains separate instances of both environments
- Tracks which environment each parallel env is currently using
- Randomly switches environments on episode completion
- Ensures both environments step forward together to maintain synchronization


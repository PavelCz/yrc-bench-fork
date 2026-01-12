#!/usr/bin/env python
"""Generate test evaluation info for a new environment.

This script evaluates the strong agent on a test environment to collect statistics
needed for agent switching cost calculations. The stats are saved to
YRC/core/test_eval_info.json.

NOTE: If you are NOT using the agent switching cost feature (i.e., strong_query_cost_ratio
and switch_agent_cost_ratio are both 0 in your config), these stats are not relevant
and are only needed for compatibility. In that case, you can simply copy stats from
a similar environment (e.g., maze_afh can use maze's stats since they share dynamics).

Usage:
    python scripts/generate_test_eval_info.py -c configs/procgen_ood.yaml -en maze_afh

The script will:
1. Load the config and create environments
2. Run the strong agent on the test environment
3. Collect episode statistics (length, reward, etc.)
4. Save the stats to YRC/core/test_eval_info.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from YRC.core.configs import make_config, set_global_variable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test evaluation info for a new environment"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "-en", "--env_name", type=str, required=True, help="Environment name"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=128,
        help="Number of episodes to evaluate (default: 128)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32)",
    )
    return parser.parse_args()


def evaluate_strong_agent(env, strong_agent, num_episodes: int) -> dict:
    """Evaluate strong agent and collect statistics."""
    episode_lengths = []
    episode_rewards = []

    obs = env.reset()
    current_rewards = np.zeros(env.num_envs)
    current_lengths = np.zeros(env.num_envs, dtype=int)

    episodes_completed = 0

    while episodes_completed < num_episodes:
        action = strong_agent.act(obs, greedy=True)
        obs, reward, done, info = env.step(action)

        current_rewards += reward
        current_lengths += 1

        for i in range(env.num_envs):
            if done[i]:
                episode_lengths.append(current_lengths[i])
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0
                current_lengths[i] = 0
                episodes_completed += 1

                if episodes_completed % 20 == 0:
                    logging.info(f"Completed {episodes_completed}/{num_episodes} episodes")

                if episodes_completed >= num_episodes:
                    break

    return {
        "steps": int(sum(episode_lengths)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_min": int(np.min(episode_lengths)),
        "episode_length_max": int(np.max(episode_lengths)),
        "reward_mean": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "env_reward_mean": 0.0,
        "env_reward_std": 0.0,
        "action_1_frac": 0.0,
    }


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Load config
    config = make_config(args.config)

    # Override environment name
    config.environment.common.env_name = args.env_name

    # Set global variables
    set_global_variable("benchmark", config.general.benchmark)

    benchmark = config.general.benchmark
    env_name = args.env_name

    logging.info(f"Generating test eval info for {benchmark}/{env_name}")

    # Import environment creation functions
    import importlib

    module = importlib.import_module(f"YRC.envs.{benchmark}")
    create_env = getattr(module, "create_env")
    load_policy = getattr(module, "load_policy")

    # Create test environment
    logging.info("Creating test environment...")
    test_env = create_env("test", config.environment)

    # Load strong agent
    logging.info("Loading strong agent...")
    strong_agent = load_policy(config.agents.strong, test_env)

    # Evaluate
    logging.info(f"Evaluating strong agent for {args.num_episodes} episodes...")
    stats = evaluate_strong_agent(test_env, strong_agent, args.num_episodes)

    test_env.close()

    # Load existing data
    test_eval_info_path = Path("YRC/core/test_eval_info.json")
    with open(test_eval_info_path) as f:
        data = json.load(f)

    # Check if entry already exists
    if env_name in data.get(benchmark, {}):
        logging.warning(f"Entry for {benchmark}/{env_name} already exists!")
        logging.warning("Existing stats:")
        logging.warning(json.dumps(data[benchmark][env_name], indent=2))
        logging.warning("New stats:")
        logging.warning(json.dumps(stats, indent=2))

        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            logging.info("Aborted.")
            return

    # Save stats
    if benchmark not in data:
        data[benchmark] = {}
    data[benchmark][env_name] = stats

    # Backup existing file
    backup_path = Path("YRC/core/backup_test_eval_info.json")
    with open(backup_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save updated file
    with open(test_eval_info_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"Saved stats for {benchmark}/{env_name}:")
    logging.info(json.dumps(stats, indent=2))
    logging.info(f"Stats saved to {test_eval_info_path}")


if __name__ == "__main__":
    main()

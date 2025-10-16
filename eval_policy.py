"""
Evaluation script to calculate mean return of an underlying agent policy.

This script evaluates a single agent policy (e.g., weak or strong SB3 agent) directly,
not a coordination policy.

Usage:
    python eval_policy.py -c <config_file> --model_file <path_to_model> [options]

Examples:
    # Evaluate a specific model (YRC loader)
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth \
        -num_rollouts 100
    
    # Evaluate with alternative loader
    export COINRUN_BG_EXTRAHARD=/path/to/model.pth
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file $COINRUN_BG_EXTRAHARD \
        -loader_type alternative \
        -num_rollouts 100

Options:
    -c, --config: Path to YAML config file (required)
    --model_file: Path to the model checkpoint to evaluate (required)
    -loader_type: Loader type - 'yrc' (default) or 'alternative' (train-procgen-pytorch-backgrounds)
    -num_rollouts: Number of episodes to evaluate (default: from config)
    -num_envs: Number of parallel environments
    -seed: Random seed
    See flags.py for more options.

Output:
    - Prints mean return statistics to console
    - Saves results to <experiment_dir>/policy_eval_results.json
"""

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.configs.global_configs import get_global_variable
from pathlib import Path
import importlib
import numpy as np
import json
import logging
import sys
import os
import torch


def load_policy_alternative(model_file, env, device):
    """
    Load policy using the train-procgen-pytorch-backgrounds method.
    This matches the loading approach in render.py from that codebase.
    """
    # Add the train-procgen-pytorch-backgrounds directory to the path
    train_procgen_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../train-procgen-pytorch-backgrounds")
    )
    if train_procgen_path not in sys.path:
        sys.path.insert(0, train_procgen_path)

    try:
        from common.model import NatureModel, ImpalaModel
        from common.policy import CategoricalPolicy
        import gym
    except ImportError as e:
        raise ImportError(
            f"Could not import from train-procgen-pytorch-backgrounds. "
            f"Make sure the directory exists at {train_procgen_path}. Error: {e}"
        )

    logging.info("Using alternative loader (train-procgen-pytorch-backgrounds)")

    # Get environment information
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Default to impala architecture (can be made configurable if needed)
    architecture = "impala"
    logging.info(f"Using {architecture} architecture")

    # Create model
    if architecture == "nature":
        model = NatureModel(in_channels=in_channels)
    elif architecture == "impala":
        model = ImpalaModel(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Create policy (non-recurrent by default)
    recurrent = False
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError("Only discrete action spaces supported")

    policy.to(device)

    # Load checkpoint
    logging.info(f"Loading checkpoint from {model_file}")
    if not Path(model_file).is_file():
        raise FileNotFoundError(f"Model file {Path(model_file).absolute()} not found")
    checkpoint = torch.load(model_file, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    # Create a simple wrapper with an act method
    class PolicyWrapper:
        def __init__(self, policy, device):
            self.policy = policy
            self.device = device

        def eval(self):
            self.policy.eval()

        def act(self, obs, greedy=False):
            # Convert obs to tensor if needed
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs).to(self.device)
            elif obs.device != self.device:
                obs = obs.to(self.device)

            with torch.no_grad():
                # Create dummy hidden state and masks for non-recurrent policy
                batch_size = obs.shape[0]
                hidden_state = torch.zeros(batch_size, 1).to(self.device)
                masks = torch.ones(batch_size, 1).to(self.device)

                # Get action distribution
                dist, value, _ = self.policy(obs, hidden_state, masks)

                if greedy:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()

                return action.cpu().numpy()

    return PolicyWrapper(policy, device)


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    # Check that model_file is provided
    if not hasattr(args, "model_file") or args.model_file is None:
        raise ValueError(
            "Must provide --model_file argument with path to model checkpoint.\n"
            "Example: python eval_policy.py -c config.yaml --model_file /path/to/model.pth"
        )

    model_file = args.model_file
    loader_type = getattr(args, "loader_type", "yrc")
    logging.info(f"Loading policy from: {model_file} using loader: {loader_type}")

    # Create a single environment (we only need one for loading the policy)
    # We'll use the raw environment creation to avoid loading coordination agents
    benchmark = get_global_variable("benchmark")
    module = importlib.import_module(f"YRC.envs.{benchmark}")

    # Create just the train environment for policy loading
    create_env_fn = getattr(module, "create_env")
    train_env = create_env_fn("train", config.environment)

    # Load the policy using the specified loader
    if loader_type == "yrc":
        logging.info(f"Loading policy using module YRC.envs.{benchmark}")
        load_policy_fn = getattr(module, "load_policy")
        policy = load_policy_fn(model_file, train_env)
        policy.eval()
    elif loader_type == "alternative":
        device = get_global_variable("device")
        policy = load_policy_alternative(model_file, train_env, device)
    else:
        raise ValueError(
            f"Unknown loader_type: {loader_type}. Must be 'yrc' or 'alternative'"
        )

    logging.info("Policy loaded successfully")

    # Number of episodes to evaluate
    num_episodes = (
        config.algorithm.num_rollouts
        if hasattr(config.algorithm, "num_rollouts")
        else 100
    )

    # Determine which environment split to use (train/test/val)
    eval_split = getattr(args, "eval_split", "test")
    logging.info(f"Creating {eval_split} environment for evaluation")

    # Create the evaluation environment
    env = create_env_fn(eval_split, config.environment)

    logging.info(
        f"Evaluating policy on {num_episodes} episodes using {eval_split} environment"
    )
    logging.info(f"Number of parallel environments: {env.num_envs}")

    # Run evaluation
    returns = rollout_and_get_returns(policy, env, num_episodes)

    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    median_return = np.median(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)

    # Print results
    print("\n" + "=" * 60)
    print("Policy Evaluation Results")
    print("=" * 60)
    print(f"Number of episodes: {len(returns)}")
    print(f"Mean return: {mean_return:.4f}")
    print(f"Std return: {std_return:.4f}")
    print(f"Median return: {median_return:.4f}")
    print(f"Min return: {min_return:.4f}")
    print(f"Max return: {max_return:.4f}")
    print("=" * 60 + "\n")

    # Save results
    save_dir = Path(str(get_global_variable("experiment_dir")))
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "num_episodes": len(returns),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "median_return": float(median_return),
        "min_return": float(min_return),
        "max_return": float(max_return),
        "all_returns": [float(r) for r in returns],
        "eval_split": eval_split,
        "model_file": model_file,
        "loader_type": loader_type,
    }

    results_path = save_dir / "policy_eval_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_path}")


def rollout_and_get_returns(policy, env, num_episodes):
    """
    Rollout the policy on the environment and collect episode returns.

    Args:
        policy: The policy to evaluate (underlying agent)
        env: The raw environment to evaluate on
        num_episodes: Number of episodes to run

    Returns:
        List of episode returns
    """
    assert num_episodes % env.num_envs == 0, (
        f"num_episodes ({num_episodes}) must be divisible by num_envs ({env.num_envs})"
    )

    returns = []
    num_completed = 0
    target_episodes = num_episodes

    # Track cumulative reward for each parallel environment
    cumulative_rewards = [0.0] * env.num_envs

    # Reset environment
    obs = env.reset()

    logging.info(f"Starting rollouts for {num_episodes} episodes...")

    while num_completed < target_episodes:
        # Get action from policy (pass observation directly to underlying agent)
        action = policy.act(obs, greedy=True)

        # Step environment
        obs, reward, done, info = env.step(action)

        # Accumulate rewards
        for i in range(env.num_envs):
            cumulative_rewards[i] += reward[i]

            # If episode is done, save the return and reset
            if done[i]:
                if num_completed < target_episodes:
                    returns.append(cumulative_rewards[i])
                    num_completed += 1

                    if num_completed % 10 == 0 or num_completed == target_episodes:
                        logging.info(
                            f"Completed {num_completed}/{target_episodes} episodes. "
                            f"Current mean return: {np.mean(returns):.4f}"
                        )

                cumulative_rewards[i] = 0.0

    return returns


if __name__ == "__main__":
    main()

"""
Evaluation script to calculate mean return of an underlying agent policy.

This script evaluates a single agent policy (e.g., weak or strong SB3 agent) directly,
not a coordination policy.

Usage:
    python eval_policy.py -c <config_file> --model_file <path_to_model> \
        -experiment_group <group_name> [options]

Examples:
    # Evaluate a specific model
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth \
        -experiment_group policy_eval \
        -num_rollouts 8
    
    # With environment variable
    export COINRUN_BG_EXTRAHARD=/path/to/model.pth
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file $COINRUN_BG_EXTRAHARD \
        -experiment_group policy_eval \
        -num_rollouts 8
    
    # With video recording to disk
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file model.pth \
        -experiment_group policy_eval \
        -num_rollouts 8 \
        -video_logging_mode folder \
        -video_output_folder ./videos \
        -video_episodes_to_collect 10
    
    # With video logging to wandb
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file model.pth \
        -experiment_group policy_eval \
        -num_rollouts 8 \
        -video_logging_mode wandb \
        -video_episodes_to_collect 10 \
        -wandb_project my_project
    
    # With video logging to both disk and wandb
    python eval_policy.py -c configs/procgen_threshold.yaml \
        --model_file model.pth \
        -experiment_group policy_eval \
        -num_rollouts 8 \
        -video_logging_mode both \
        -video_output_folder ./videos \
        -video_episodes_to_collect 10

Options:
    -c, --config: Path to YAML config file (required)
    --model_file: Path to the model checkpoint to evaluate (required)
    -experiment_group: Required eval output group name
    -num_rollouts: Number of episodes to evaluate (default: from config)
    -num_envs: Number of parallel environments
    -eval_split: One of train, val_sim, val_true, test (default: test)
    -seed: Random seed
    -greedy: Use greedy action selection (default: True)
    -video_logging_mode: Video logging mode: 'folder', 'wandb', 'both', 'none' (default)
    -video_output_folder: Folder path for saving videos (default: <eval_run_dir>/videos)
    -video_episodes_to_collect: Number of episodes to save as videos (default: 0)
    -wandb_project: Weights & Biases project name (for wandb logging)
    -wandb_mode: wandb mode: 'online', 'offline', 'disabled' (default: 'online')
    See flags.py for more options.

Output:
    - Prints mean return statistics to console
    - Saves results to <eval_run_dir>/policy_eval_results.json
    - Optionally saves episode videos to disk and/or Weights & Biases
"""

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.configs.global_configs import get_global_variable
from pathlib import Path
import importlib
import numpy as np
import json
import logging
from typing import List, Dict, Optional
import wandb
from pytorch_lightning.loggers import WandbLogger
from YRC.core.video_utils import (
    VideoProcessor,
    save_video_to_folder,
    resolve_video_output_folder,
)


# Video configuration
VIDEO_CONFIG = {
    "fps": 10,
    "final_frame_repetitions": 10,
    "score_bar_height_ratio": 0.0,  # No score bar for simple policy evaluation
    "score_bar_bg_color": 64,  # Dark gray
    "text_padding": 5,
    "normal_color": [0, 255, 0],  # Green
    "ood_color": [255, 0, 0],  # Red
    "text_color": [255, 255, 255],  # White
    "outline_color": [0, 0, 0],  # Black
    "min_output_size": 512,  # Minimum output size for videos
}


def load_eval_level_seeds(config) -> Optional[List[int]]:
    """Load fixed evaluation seeds from config.environment.level_seeds_file.

    The seed files generated for this repo contain separate splits. `scripts/run_eval.py`
    passes these files to `eval_afhp.py`, which uses the `ood_eval` split for test-time
    evaluation. We mirror that behavior here so standalone policy evaluation can run on
    the same fixed levels.
    """
    level_seeds_file = getattr(config.environment, "level_seeds_file", None)
    if level_seeds_file is None:
        return None

    logging.info(f"Loading level seeds from {level_seeds_file}")
    with open(level_seeds_file) as f:
        seeds_data = json.load(f)

    level_seeds = seeds_data["seeds"].get("ood_eval") or None
    if level_seeds is not None:
        logging.info(f"Loaded {len(level_seeds)} ood_eval seeds for evaluation")
    else:
        logging.warning("No ood_eval seeds found in level seeds file; using default env seeds")

    return level_seeds


def summarize_returns(returns: List[float]) -> Dict[str, Optional[float]]:
    """Compute summary stats for a list of episodic returns."""
    if len(returns) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    returns_np = np.asarray(returns, dtype=np.float32)
    return {
        "count": int(len(returns)),
        "mean": float(np.mean(returns_np)),
        "std": float(np.std(returns_np)),
        "median": float(np.median(returns_np)),
        "min": float(np.min(returns_np)),
        "max": float(np.max(returns_np)),
    }


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
    logging.info(f"Loading policy from: {model_file}")

    # Create a single environment (we only need one for loading the policy)
    # We'll use the raw environment creation to avoid loading coordination agents
    benchmark = get_global_variable("benchmark")
    module = importlib.import_module(f"YRC.envs.{benchmark}")

    # Create just the train environment for policy loading
    create_env_fn = getattr(module, "create_env")
    train_env = create_env_fn("train", config.environment)

    print(f"Loading policy using module YRC.envs.{benchmark}")
    # Load the policy directly using the environment-specific load function
    load_policy_fn = getattr(module, "load_policy")
    policy = load_policy_fn(model_file, train_env)
    policy.eval()

    logging.info("Policy loaded successfully")

    # Number of episodes to evaluate
    num_episodes = (
        config.algorithm.num_rollouts
        if hasattr(config.algorithm, "num_rollouts")
        else 100
    )

    # Determine which environment split to use (train/test/val)
    eval_split = getattr(args, "eval_split", None) or "test"
    logging.info(f"Creating {eval_split} environment for evaluation")

    # If a level seeds file is provided, mirror eval_afhp.py and evaluate on the fixed
    # ood_eval split in sequential order.
    level_seeds = load_eval_level_seeds(config)

    # Create the evaluation environment
    if level_seeds is not None:
        env = create_env_fn(
            eval_split,
            config.environment,
            level_seeds=level_seeds,
            level_seeds_mode="sequential",
        )
    else:
        env = create_env_fn(eval_split, config.environment)

    logging.info(
        f"Evaluating policy on {num_episodes} episodes using {eval_split} environment"
    )
    logging.info(f"Number of parallel environments: {env.num_envs}")

    # Get greedy flag from args (default to True if not set)
    greedy = (
        getattr(config.policy, "greedy", True) if hasattr(config, "policy") else True
    )
    logging.info(f"Using greedy action selection: {greedy}")

    # Use eval_run_dir in eval mode to keep outputs with the rest of the eval artifacts.
    if hasattr(config, "eval_run_dir") and config.eval_run_dir is not None:
        save_dir = Path(config.eval_run_dir)
    else:
        save_dir = Path(str(get_global_variable("experiment_dir")))

    # Check video collection settings
    video_logging_mode = (
        getattr(config.evaluation, "video_logging_mode", "none")
        if hasattr(config, "evaluation")
        else "none"
    )
    video_episodes_to_collect = (
        getattr(config.evaluation, "video_episodes_to_collect", 0)
        if hasattr(config, "evaluation")
        else 0
    )

    should_collect_videos = (
        video_logging_mode in ["folder", "wandb", "both"]
        and video_episodes_to_collect > 0
    )

    if should_collect_videos:
        logging.info(
            f"Video collection enabled: will collect {video_episodes_to_collect} episodes"
        )
        logging.info(f"Video logging mode: {video_logging_mode}")

    # Initialize wandb if needed for video logging
    wandb_logger = None
    if video_logging_mode in ["wandb", "both"] and video_episodes_to_collect > 0:
        wandb_logger = initialize_wandb_logger(config, args, save_dir)

    # Run evaluation
    returns, level_ood_gt, level_seeds, video_episodes = rollout_and_get_returns(
        policy,
        env,
        num_episodes,
        greedy=greedy,
        collect_videos=should_collect_videos,
        max_video_episodes=video_episodes_to_collect,
    )

    overall_stats = summarize_returns(returns)
    id_returns = [ret for ret, is_ood in zip(returns, level_ood_gt) if not is_ood]
    ood_returns = [ret for ret, is_ood in zip(returns, level_ood_gt) if is_ood]
    id_stats = summarize_returns(id_returns)
    ood_stats = summarize_returns(ood_returns)

    # Print results
    print("\n" + "=" * 60)
    print("Policy Evaluation Results")
    print("=" * 60)
    print(f"Number of episodes: {overall_stats['count']}")
    print(f"Mean return: {overall_stats['mean']:.4f}")
    print(f"Std return: {overall_stats['std']:.4f}")
    print(f"Median return: {overall_stats['median']:.4f}")
    print(f"Min return: {overall_stats['min']:.4f}")
    print(f"Max return: {overall_stats['max']:.4f}")
    print(f"ID episodes: {id_stats['count']}")
    if id_stats["mean"] is not None:
        print(f"ID mean return: {id_stats['mean']:.4f}")
    print(f"OOD episodes: {ood_stats['count']}")
    if ood_stats["mean"] is not None:
        print(f"OOD mean return: {ood_stats['mean']:.4f}")
    print("=" * 60 + "\n")

    # Save results
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "num_episodes": overall_stats["count"],
        "mean_return": overall_stats["mean"],
        "std_return": overall_stats["std"],
        "median_return": overall_stats["median"],
        "min_return": overall_stats["min"],
        "max_return": overall_stats["max"],
        "id_num_episodes": id_stats["count"],
        "id_mean_return": id_stats["mean"],
        "id_std_return": id_stats["std"],
        "ood_num_episodes": ood_stats["count"],
        "ood_mean_return": ood_stats["mean"],
        "ood_std_return": ood_stats["std"],
        "all_returns": [float(r) for r in returns],
        "level_ood_gt": [bool(x) for x in level_ood_gt],
        "level_seeds": level_seeds,
        "eval_split": eval_split,
        "model_file": model_file,
        "level_seeds_file": getattr(config.environment, "level_seeds_file", None),
        "num_fixed_eval_seeds": len(level_seeds) if level_seeds is not None else None,
    }

    results_path = save_dir / "policy_eval_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_path}")

    # Save videos if enabled
    if should_collect_videos and len(video_episodes) > 0:
        save_videos(video_episodes, config, save_dir, eval_split, wandb_logger)

    # Finish wandb run if it was initialized
    if wandb_logger is not None:
        wandb.finish()


def rollout_and_get_returns(
    policy, env, num_episodes, greedy=True, collect_videos=False, max_video_episodes=0
):
    """
    Rollout the policy on the environment and collect episode returns.

    Args:
        policy: The policy to evaluate (underlying agent)
        env: The raw environment to evaluate on
        num_episodes: Number of episodes to run
        greedy: Whether to use greedy action selection (default: True)
        collect_videos: Whether to collect video frames (default: False)
        max_video_episodes: Maximum number of episodes to collect videos for (default: 0)

    Returns:
        Tuple of (returns, level_ood_gt, level_seeds, video_episodes)
        - returns: List of episode returns
        - level_ood_gt: Whether each completed episode was OOD
        - level_seeds: Procgen level seed for each completed episode when available
        - video_episodes: List of collected video data (if collect_videos=True)
    """
    returns = []
    level_ood_gt = []
    level_seeds = []
    num_completed = 0
    target_episodes = num_episodes

    # Track cumulative reward for each parallel environment
    cumulative_rewards = [0.0] * env.num_envs
    # The current episode's OOD ground truth becomes available after the first step.
    current_level_ood_gt = [False] * env.num_envs

    # Video collection data structures (one per parallel env)
    video_episodes = []
    current_episodes = [[] for _ in range(env.num_envs)]
    video_episodes_collected = 0

    # Reset environment
    obs = env.reset()

    # Reset episode counter for heuristic policies at the start
    for i in range(env.num_envs):
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()

    logging.info(f"Starting rollouts for {num_episodes} episodes...")

    while num_completed < target_episodes:
        # Get action from policy (pass observation directly to underlying agent)
        action = policy.act(obs, greedy=greedy)

        # Step environment
        next_obs, reward, done, info = env.step(action)

        # Collect video frames if enabled
        if collect_videos and video_episodes_collected < max_video_episodes:
            for i in range(env.num_envs):
                # Observations are already in [0, 1] range from ScaledFloatFrame wrapper
                obs_float = obs[i].astype(np.float32)

                current_episodes[i].append(
                    {
                        "obs": obs_float,
                        "action": action[i],
                        "reward": reward[i],
                        "done": done[i],
                        "scores": None,  # No OOD scores for simple policy evaluation
                        "recons": None,  # No reconstructions
                    }
                )

        # Accumulate rewards
        for i in range(env.num_envs):
            cumulative_rewards[i] += reward[i]

            # If episode is done, save the return and reset
            if done[i]:
                if num_completed < target_episodes:
                    returns.append(cumulative_rewards[i])
                    level_ood_gt.append(bool(current_level_ood_gt[i]))
                    level_seeds.append(int(info[i].get("prev_level_seed", -1)))
                    num_completed += 1

                    if num_completed % 10 == 0 or num_completed == target_episodes:
                        logging.info(
                            f"Completed {num_completed}/{target_episodes} episodes. "
                            f"Current mean return: {np.mean(returns):.4f}"
                        )

                # Save video episode if we're still collecting
                if (
                    collect_videos
                    and video_episodes_collected < max_video_episodes
                    and len(current_episodes[i]) > 0
                ):
                    video_episodes.append(
                        {
                            "frames": current_episodes[i],
                            "return": cumulative_rewards[i],
                            "episode_idx": video_episodes_collected,
                        }
                    )
                    video_episodes_collected += 1
                    logging.info(
                        f"Collected video {video_episodes_collected}/{max_video_episodes}"
                    )

                # Reset for next episode
                current_episodes[i] = []
                cumulative_rewards[i] = 0.0

                # Reset episode counter for heuristic policies
                if hasattr(policy, "reset_episode"):
                    policy.reset_episode()

            # Gym3 auto-resets immediately, so after processing any done episode we
            # update the OOD label from the current info dict for the next episode.
            if "randomize_goal" in info[i]:
                current_level_ood_gt[i] = bool(info[i]["randomize_goal"])

        obs = next_obs

    return returns, level_ood_gt, level_seeds, video_episodes


def initialize_wandb_logger(config, args, save_dir: Path) -> WandbLogger:
    """
    Initialize Weights & Biases logger for video logging.

    Args:
        config: Configuration object
        args: Command line arguments
        save_dir: Directory to save wandb logs

    Returns:
        WandbLogger instance
    """
    # Prepare wandb init parameters - access config.wandb directly
    # (flags are already merged into config by config_utils.load)
    wandb_kwargs = {
        "name": config.exp_name,
        "project": config.wandb.project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "job_type": "policy_eval",
        "config": config,
    }

    if config.wandb.entity is not None:
        wandb_kwargs["entity"] = config.wandb.entity

    logging.info(
        f"Initializing wandb: project={config.wandb.project}, "
        f"name={config.exp_name}, mode={config.wandb.mode}"
    )
    exp = wandb.init(**wandb_kwargs)

    wandb_logger = WandbLogger(
        save_dir=save_dir,
        experiment=exp,
    )

    return wandb_logger


def save_videos(
    video_episodes: List[Dict],
    config,
    save_dir: Path,
    eval_split: str,
    wandb_logger: Optional[WandbLogger] = None,
):
    """
    Save collected video episodes to disk and/or wandb.

    Args:
        video_episodes: List of video episode data
        config: Configuration object
        save_dir: Directory to save videos
        eval_split: Evaluation split name (train/test/val)
        wandb_logger: Optional WandbLogger for logging to Weights & Biases
    """
    # Get video logging mode
    video_logging_mode = (
        getattr(config.evaluation, "video_logging_mode", "folder")
        if hasattr(config, "evaluation")
        else "folder"
    )

    # Determine output folder for disk saving
    output_folder = None
    split_folder = None
    if video_logging_mode in ["folder", "both"]:
        video_output_folder = (
            getattr(config.evaluation, "video_output_folder", None)
            if hasattr(config, "evaluation")
            else None
        )

        if video_output_folder is None:
            # Default to <experiment_dir>/videos
            output_folder = save_dir / "videos"
        else:
            resolved = resolve_video_output_folder(
                video_output_folder, save_dir, create_folder=True
            )
            # resolve_video_output_folder returns None only if input is None,
            # which we already checked above
            assert resolved is not None
            output_folder = resolved

        output_folder.mkdir(parents=True, exist_ok=True)

        # Create subfolder for the eval split
        split_folder = output_folder / eval_split
        split_folder.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving {len(video_episodes)} videos to {split_folder}")

    # Create video processor
    processor = VideoProcessor(VIDEO_CONFIG)

    for video_data in video_episodes:
        frames = video_data["frames"]
        episode_return = video_data["return"]
        episode_idx = video_data["episode_idx"]

        # Extract observations
        observations = [frame["obs"] for frame in frames]

        # Create video
        video = np.stack(observations, axis=0)

        # Add repeated frames for smoother ending
        video = processor.add_repeated_frames(video)

        # Normalize to 0-255 range
        video = video * 255
        video = video.astype(np.uint8)

        # Generate filename and caption
        filename = f"episode_{episode_idx:03d}_return_{episode_return:.2f}"
        caption = f"Episode {episode_idx} - Split: {eval_split} - Return: {episode_return:.2f}"

        # Save to disk if needed
        if video_logging_mode in ["folder", "both"]:
            save_video_to_folder(
                video,
                split_folder,
                filename,
                VIDEO_CONFIG,
                caption=caption,
            )

        # Log to wandb if needed
        if video_logging_mode in ["wandb", "both"] and wandb_logger is not None:
            # Create video key with category
            video_key = f"policy_eval/{eval_split}/episode_{episode_return:.2f}"

            # Log video to wandb
            wandb_logger.experiment.log(
                {
                    video_key: wandb.Video(
                        video,
                        fps=VIDEO_CONFIG["fps"],
                        format="gif",
                        caption=caption,
                    ),
                }
            )

    if video_logging_mode in ["folder", "both"]:
        logging.info(f"Successfully saved {len(video_episodes)} videos to disk")
    if video_logging_mode in ["wandb", "both"]:
        logging.info(f"Successfully logged {len(video_episodes)} videos to wandb")


if __name__ == "__main__":
    main()

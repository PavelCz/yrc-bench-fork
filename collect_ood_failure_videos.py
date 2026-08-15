"""Collect weak-policy videos of Heist ground-truth OOD timeouts.

Unlike AFHP evaluation video logging, this script keeps an episode only when it
is out of distribution and ends without completing the level. It stops after
saving the requested number of matching videos, or after a search budget of
rollouts.

Example:
    python collect_ood_failure_videos.py \\
        -c configs/eval/heist/max_prob.yaml \\
        -n heist_ood_timeout_videos \\
        -experiment_group tmlr-heist02-ood-timeout-videos \\
        -en heist_afh \\
        --model_file /path/to/weak.pth \\
        -level_seeds_file /path/to/0.json \\
        -eval_split test \\
        -num_rollouts 256 \\
        -video_episodes_to_collect 16 \\
        -video_logging_mode folder \\
        -wandb_mode disabled \\
        -greedy false
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.video_utils import (
    VideoProcessor,
    resolve_video_output_folder,
    save_video_to_folder,
)
from YRC.envs.procgen.heist_metrics import HEIST_ENV_NAME, extract_heist_episode_data


VIDEO_CONFIG = {
    "fps": 10,
    "final_frame_repetitions": 10,
    "min_output_size": 512,
}


def is_ood_timeout(*, is_ood: bool, level_complete: bool) -> bool:
    """Return whether an episode is a ground-truth OOD timeout."""
    return bool(is_ood) and not bool(level_complete)


def ood_timeout_video_filename(
    saved_idx: int,
    *,
    level_seed: int,
    keys_collected: int,
    num_keys: int,
    chests_opened: int,
    total_chests: int,
    episode_return: float,
) -> str:
    """Return a stable filename for one saved OOD-timeout video."""
    return (
        f"ood_timeout_{saved_idx:03d}"
        f"_seed{level_seed}"
        f"_keys{keys_collected}-{num_keys}"
        f"_chests{chests_opened}-{total_chests}"
        f"_ret{episode_return:.2f}"
    )


def ood_timeout_video_caption(
    *,
    saved_idx: int,
    level_seed: int,
    keys_collected: int,
    num_keys: int,
    chests_opened: int,
    total_chests: int,
    episode_return: float,
    episode_length: int,
) -> str:
    """Return a short caption describing one saved OOD-timeout episode."""
    return (
        f"OOD timeout {saved_idx:03d}: seed={level_seed}, "
        f"keys={keys_collected}/{num_keys}, "
        f"chests={chests_opened}/{total_chests}, "
        f"return={episode_return:.2f}, length={episode_length}"
    )


def episode_randomize_goal(
    info: Mapping[str, Any], done: bool, current_value: bool
) -> bool:
    """Return the OOD label for the episode that just ended."""
    if done and "prev_level/randomize_goal" in info:
        return bool(info["prev_level/randomize_goal"])
    return bool(current_value)


def load_eval_level_seeds(config) -> Optional[List[int]]:
    """Load the ood_eval split from config.environment.level_seeds_file."""
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
        logging.warning(
            "No ood_eval seeds found in level seeds file; using default env seeds"
        )

    return level_seeds


def _copy_human_frame(info: Mapping[str, Any]) -> Optional[np.ndarray]:
    rgb = info.get("rgb")
    if rgb is None:
        return None
    return np.asarray(rgb).copy()


def _resolve_greedy(config) -> bool:
    if hasattr(config, "policy") and getattr(config.policy, "greedy", None) is not None:
        return bool(config.policy.greedy)
    return False


def save_ood_timeout_video(
    frames: Sequence[Mapping[str, Any]],
    *,
    output_folder: Path,
    filename: str,
    caption: str,
    include_human_view: bool,
) -> Path:
    """Encode one collected episode and write it as a GIF."""
    processor = VideoProcessor(VIDEO_CONFIG)
    observations = [np.asarray(frame["obs"], dtype=np.float32) for frame in frames]
    video = np.stack(observations, axis=0)
    video = processor.add_repeated_frames(video)
    video = (np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)

    human_obs = [frame.get("human_obs") for frame in frames]
    if include_human_view and any(frame is not None for frame in human_obs):
        video = processor.combine_agent_and_human_views(video, human_obs)
    else:
        video = processor.upscale_video(video, VIDEO_CONFIG["min_output_size"])

    save_video_to_folder(video, output_folder, filename, VIDEO_CONFIG, caption=caption)
    return output_folder / f"{filename}.gif"


def collect_ood_timeout_videos(
    policy,
    env,
    *,
    max_search_episodes: int,
    max_videos: int,
    greedy: bool,
    output_folder: Path,
    include_human_view: bool,
) -> Dict[str, Any]:
    """Roll out until enough OOD timeouts are saved, or the search budget ends."""
    if max_videos <= 0:
        raise ValueError(f"max_videos must be positive, got {max_videos}")
    if max_search_episodes <= 0:
        raise ValueError(
            f"max_search_episodes must be positive, got {max_search_episodes}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    returns: List[float] = []
    current_rewards = [0.0] * env.num_envs
    current_lengths = [0] * env.num_envs
    current_level_ood_gt = [False] * env.num_envs
    recording_episode = [True] * env.num_envs
    current_episodes: List[List[Dict[str, Any]]] = [[] for _ in range(env.num_envs)]
    prev_info: List[Dict[str, Any]] = [{} for _ in range(env.num_envs)]
    seeds_exhausted = np.zeros(env.num_envs, dtype=bool)

    saved_videos: List[Dict[str, Any]] = []
    num_completed = 0
    num_ood = 0
    num_ood_timeouts = 0

    obs = env.reset()
    for env_idx in range(env.num_envs):
        if hasattr(policy, "reset_episode"):
            policy.reset_episode(env_idx)

    logging.info(
        f"Searching up to {max_search_episodes} episodes for "
        f"{max_videos} OOD-timeout videos"
    )

    while num_completed < max_search_episodes and len(saved_videos) < max_videos:
        action = policy.act(obs, greedy=greedy)
        next_obs, reward, done, info = env.step(action)

        for env_idx in range(env.num_envs):
            if seeds_exhausted[env_idx]:
                continue

            current_rewards[env_idx] += float(reward[env_idx])
            current_lengths[env_idx] += 1

            still_need_videos = len(saved_videos) < max_videos
            if still_need_videos and recording_episode[env_idx]:
                current_episodes[env_idx].append(
                    {
                        "obs": np.asarray(obs[env_idx], dtype=np.float32).copy(),
                        "human_obs": _copy_human_frame(prev_info[env_idx]),
                        "action": action[env_idx],
                        "reward": reward[env_idx],
                        "done": bool(done[env_idx]),
                    }
                )

            if done[env_idx]:
                if (
                    num_completed >= max_search_episodes
                    or len(saved_videos) >= max_videos
                ):
                    current_episodes[env_idx] = []
                    current_rewards[env_idx] = 0.0
                    current_lengths[env_idx] = 0
                    recording_episode[env_idx] = True
                    if hasattr(policy, "reset_episode"):
                        policy.reset_episode(env_idx)
                    continue

                episode_return = current_rewards[env_idx]
                episode_length = current_lengths[env_idx]
                is_ood = episode_randomize_goal(
                    info[env_idx], True, current_level_ood_gt[env_idx]
                )
                heist_data = extract_heist_episode_data(info[env_idx])
                level_seed = int(info[env_idx].get("prev_level_seed", -1))
                returns.append(episode_return)
                num_completed += 1
                if is_ood:
                    num_ood += 1

                keep = is_ood_timeout(
                    is_ood=is_ood, level_complete=heist_data["level_complete"]
                )
                if keep:
                    num_ood_timeouts += 1

                if keep and still_need_videos and current_episodes[env_idx]:
                    saved_idx = len(saved_videos)
                    filename = ood_timeout_video_filename(
                        saved_idx,
                        level_seed=level_seed,
                        keys_collected=heist_data["keys_collected"],
                        num_keys=heist_data["num_keys"],
                        chests_opened=heist_data["chests_opened"],
                        total_chests=heist_data["total_chests"],
                        episode_return=episode_return,
                    )
                    caption = ood_timeout_video_caption(
                        saved_idx=saved_idx,
                        level_seed=level_seed,
                        keys_collected=heist_data["keys_collected"],
                        num_keys=heist_data["num_keys"],
                        chests_opened=heist_data["chests_opened"],
                        total_chests=heist_data["total_chests"],
                        episode_return=episode_return,
                        episode_length=episode_length,
                    )
                    video_path = save_ood_timeout_video(
                        current_episodes[env_idx],
                        output_folder=output_folder,
                        filename=filename,
                        caption=caption,
                        include_human_view=include_human_view,
                    )
                    record = {
                        "filename": filename,
                        "path": str(video_path),
                        "level_seed": level_seed,
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        **heist_data,
                    }
                    saved_videos.append(record)
                    logging.info(
                        f"Saved OOD-timeout video {len(saved_videos)}/{max_videos}: "
                        f"{filename}"
                    )

                if num_completed % 10 == 0 or len(saved_videos) >= max_videos:
                    logging.info(
                        f"Searched {num_completed}/{max_search_episodes} episodes "
                        f"({num_ood} OOD, {num_ood_timeouts} OOD timeouts); "
                        f"saved {len(saved_videos)}/{max_videos} videos"
                    )

                current_episodes[env_idx] = []
                current_rewards[env_idx] = 0.0
                current_lengths[env_idx] = 0
                recording_episode[env_idx] = True
                if hasattr(policy, "reset_episode"):
                    policy.reset_episode(env_idx)

            if "randomize_goal" in info[env_idx]:
                current_level_ood_gt[env_idx] = bool(info[env_idx]["randomize_goal"])
                if not current_level_ood_gt[env_idx]:
                    recording_episode[env_idx] = False
                    current_episodes[env_idx] = []

            if info[env_idx].get("seeds_exhausted", False):
                seeds_exhausted[env_idx] = True

            prev_info[env_idx] = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in info[env_idx].items()
            }

        obs = next_obs

        if seeds_exhausted.all():
            logging.warning(
                "All environments exhausted their level seeds after "
                f"{num_completed} completed episodes"
            )
            break

    summary = {
        "num_search_episodes": num_completed,
        "num_ood_episodes": num_ood,
        "num_ood_timeouts": num_ood_timeouts,
        "num_videos_saved": len(saved_videos),
        "requested_videos": max_videos,
        "max_search_episodes": max_search_episodes,
        "videos": saved_videos,
        "all_returns": [float(value) for value in returns],
    }
    return summary


def main() -> int:
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    env_name = config.environment.common.env_name
    if env_name != HEIST_ENV_NAME:
        raise ValueError(
            f"collect_ood_failure_videos.py only supports heist_afh, got {env_name!r}"
        )

    if not hasattr(args, "model_file") or args.model_file is None:
        raise ValueError(
            "Must provide --model_file with the weak-policy checkpoint to roll out"
        )

    model_file = args.model_file
    max_videos = int(getattr(config.evaluation, "video_episodes_to_collect", 16) or 16)
    max_search_episodes = int(getattr(config.algorithm, "num_rollouts", 256) or 256)
    include_human_view = bool(getattr(config.evaluation, "include_human_view", True))
    greedy = _resolve_greedy(config)
    eval_split = getattr(args, "eval_split", None) or "test"

    logging.info(f"Loading policy from: {model_file}")
    benchmark = get_global_variable("benchmark")
    module = importlib.import_module(f"YRC.envs.{benchmark}")
    create_env_fn = getattr(module, "create_env")
    load_policy_fn = getattr(module, "load_policy")

    train_env = create_env_fn("train", config.environment)
    policy = load_policy_fn(model_file, train_env)
    policy.eval()

    level_seeds = load_eval_level_seeds(config)
    if level_seeds is not None:
        env = create_env_fn(
            eval_split,
            config.environment,
            level_seeds=level_seeds,
            level_seeds_mode="sequential",
        )
    else:
        env = create_env_fn(eval_split, config.environment)

    if hasattr(config, "eval_run_dir") and config.eval_run_dir is not None:
        save_dir = Path(config.eval_run_dir)
    else:
        save_dir = Path(str(get_global_variable("experiment_dir")))
    save_dir.mkdir(parents=True, exist_ok=True)

    video_output_folder = getattr(config.evaluation, "video_output_folder", None)
    if video_output_folder is None:
        output_folder = save_dir / "videos" / "ood_timeout"
    else:
        resolved = resolve_video_output_folder(
            video_output_folder, save_dir, create_folder=True
        )
        assert resolved is not None
        output_folder = resolved / "ood_timeout"
    output_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Eval split: {eval_split}")
    logging.info(f"Parallel envs: {env.num_envs}")
    logging.info(f"Greedy actions: {greedy}")
    logging.info(f"Human view: {include_human_view}")
    logging.info(f"Video folder: {output_folder}")

    summary = collect_ood_timeout_videos(
        policy,
        env,
        max_search_episodes=max_search_episodes,
        max_videos=max_videos,
        greedy=greedy,
        output_folder=output_folder,
        include_human_view=include_human_view,
    )
    summary.update(
        {
            "model_file": model_file,
            "eval_split": eval_split,
            "greedy": greedy,
            "include_human_view": include_human_view,
            "level_seeds_file": getattr(config.environment, "level_seeds_file", None),
            "output_folder": str(output_folder),
        }
    )

    results_path = save_dir / "ood_timeout_videos.json"
    with results_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Wrote video manifest to {results_path}")
    print(f"Saved {summary['num_videos_saved']}/{max_videos} OOD-timeout videos")
    print(f"Searched {summary['num_search_episodes']} episodes")
    print(f"Videos: {output_folder}")
    print(f"Manifest: {results_path}")

    if summary["num_videos_saved"] == 0:
        logging.error("No OOD-timeout videos were collected")
        return 1
    if summary["num_videos_saved"] < max_videos:
        logging.warning(
            f"Only collected {summary['num_videos_saved']}/{max_videos} "
            "OOD-timeout videos before the search budget ended"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

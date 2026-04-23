from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import wandb

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_core
import YRC.core.policy as policy_factory
from YRC.core.eval_script_utils import init_eval_wandb_run, save_npz_results
from YRC.coverage.coverage_search import update_policy_params
from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy


RETURN_ATOL = 1e-5


@dataclass
class PointRecord:
    index: int
    threshold: float
    split: str
    level_seeds: List[int]
    level_ood_pred: List[bool]
    first_help_timesteps: List[Optional[int]]
    raw_returns: List[float]


@dataclass
class SanityRolloutResult:
    level_seeds: List[int]
    level_ood_pred: List[bool]
    first_help_timesteps: List[Optional[int]]
    raw_returns: List[float]
    pre_help_actions_by_seed: Dict[int, List[int]]

    @property
    def seed_to_return(self) -> Dict[int, float]:
        return dict(zip(self.level_seeds, self.raw_returns))


def resolve_action_greedy(config):
    policy_config = getattr(config, "policy", None)
    greedy = (
        getattr(policy_config, "greedy", None) if policy_config is not None else None
    )
    if greedy is not None:
        return bool(greedy)

    coord_env_config = getattr(config, "coord_env", None)
    coord_env_greedy = (
        getattr(coord_env_config, "act_greedy", None)
        if coord_env_config is not None
        else None
    )
    return bool(coord_env_greedy) if coord_env_greedy is not None else False


def get_info_value(info, key, index, default=None):
    if isinstance(info, list):
        if index >= len(info) or not isinstance(info[index], dict):
            return default
        return info[index].get(key, default)

    if isinstance(info, dict):
        value = info.get(key, default)
        if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            try:
                return value[index]
            except (IndexError, KeyError, TypeError):
                return value
        return value

    return default


def get_info_values(info, key, num_envs, default=None):
    return [get_info_value(info, key, i, default) for i in range(num_envs)]


def get_completed_level_seed(info, index):
    level_seed = get_info_value(info, "prev_level_seed", index)
    if level_seed is None:
        level_seed = get_info_value(info, "level_seed", index)

    if level_seed is None:
        return None
    try:
        level_seed = int(level_seed)
    except (TypeError, ValueError):
        return None
    return None if level_seed < 0 else level_seed


def get_seeds_exhausted(info, num_envs):
    return np.array(
        [
            bool(get_info_value(info, "seeds_exhausted", i, False))
            for i in range(num_envs)
        ]
    )


def normalize_optional_timestep(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    return int(value)


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(np.mean(values))


def default_procgen_timeout(env_name: str) -> int:
    env_name = env_name.lower()
    if env_name == "maze" or env_name.startswith("maze_"):
        return 500
    return 1000


def resolve_base_timeout(config) -> int:
    configured = getattr(config.environment.common, "max_steps", None)
    if configured is not None:
        return int(configured)
    return default_procgen_timeout(config.environment.common.env_name)


def reset_timeout_cap(pre_help_steps: int, base_timeout: int) -> int:
    if pre_help_steps < 0:
        raise ValueError(f"pre_help_steps must be non-negative, got {pre_help_steps}")
    if base_timeout <= 0:
        raise ValueError(f"base_timeout must be positive, got {base_timeout}")
    return int(pre_help_steps + base_timeout)


def extract_point_record(pt_meta: Dict[str, Any], index: int, split: str = "test"):
    summary_dict = pt_meta.get("summary", {})
    if split not in summary_dict:
        available_splits = list(summary_dict.keys())
        raise ValueError(
            f"Point {index} does not contain summary split {split!r}. "
            f"Available splits: {available_splits}"
        )

    summary = summary_dict[split]
    level_seeds = [int(seed) for seed in summary.get("level_seeds", [])]
    level_ood_pred = [bool(pred) for pred in summary.get("level_ood_pred", [])]
    raw_returns = [float(ret) for ret in summary.get("raw_returns", [])]
    first_help_raw = summary.get("first_ood_timestep", None)
    if first_help_raw is None:
        first_help_timesteps = [None] * len(level_seeds)
    else:
        first_help_timesteps = [
            normalize_optional_timestep(value) for value in list(first_help_raw)
        ]

    lengths = {
        "level_seeds": len(level_seeds),
        "level_ood_pred": len(level_ood_pred),
        "first_ood_timestep": len(first_help_timesteps),
        "raw_returns": len(raw_returns),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Point {index} has misaligned summary arrays: {lengths}")

    return PointRecord(
        index=index,
        threshold=float(pt_meta.get("threshold", float("nan"))),
        split=split,
        level_seeds=level_seeds,
        level_ood_pred=level_ood_pred,
        first_help_timesteps=first_help_timesteps,
        raw_returns=raw_returns,
    )


def help_indices(record: PointRecord) -> List[int]:
    return [i for i, pred in enumerate(record.level_ood_pred) if pred]


def help_seeds(record: PointRecord) -> List[int]:
    return [record.level_seeds[i] for i in help_indices(record)]


def build_point_comparison(
    record: PointRecord,
    sanity_seed_to_return: Dict[int, float],
    strong_seed_to_return: Dict[int, float],
    reset_seed_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    indices = help_indices(record)
    seeds = [record.level_seeds[i] for i in indices]

    original_help_returns = [record.raw_returns[i] for i in indices]
    sanity_returns = [sanity_seed_to_return[seed] for seed in seeds]
    strong_returns = [strong_seed_to_return[seed] for seed in seeds]
    reset_returns = [reset_seed_results[seed]["return"] for seed in seeds]
    first_help_timesteps = [record.first_help_timesteps[i] for i in indices]

    return {
        "original_help_performance": mean_or_nan(original_help_returns),
        "sanity_performance": mean_or_nan(sanity_returns),
        "strong_performance": mean_or_nan(strong_returns),
        "reset_timeout_performance": mean_or_nan(reset_returns),
        "comparison_meta": {
            "point_idx": record.index,
            "threshold": record.threshold,
            "split": record.split,
            "help_requested": bool(seeds),
            "help_seeds": seeds,
            "first_help_timesteps": first_help_timesteps,
            "original_help_returns": original_help_returns,
            "sanity_returns": sanity_returns,
            "strong_from_start_returns": strong_returns,
            "reset_timeout_returns": reset_returns,
            "reset_timeout_caps": [
                reset_seed_results[seed]["timeout_cap"] for seed in seeds
            ],
            "assertions_passed": True,
        },
    }


def normalize_saved_device(config_dict: Dict[str, Any]) -> None:
    general = config_dict.get("general", {})
    device = general.get("device")
    if isinstance(device, str):
        if device == "cpu":
            general["device"] = -1
        elif device.startswith("cuda:"):
            general["device"] = int(device.split(":", 1)[1])
        elif device == "cuda":
            general["device"] = 0


def load_reval_config(args, npz_path: Path):
    args.eval_mode = False
    args.overwrite = True

    sibling_config = npz_path.with_name("config.json")
    if sibling_config.exists():
        print(f"Using original sibling config: {sibling_config}")
        with sibling_config.open("r") as f:
            config_dict = json.load(f)
        config_dict["eval_mode"] = False
        config_dict["overwrite"] = True
        config_dict["use_wandb"] = bool(getattr(args, "use_wandb", False))
        normalize_saved_device(config_dict)
        return config_utils.load(json.dumps(config_dict), flags=None)

    if args.config is None:
        raise ValueError(
            "Must provide -c/--config when no sibling config.json exists next to "
            f"{npz_path}"
        )
    print(
        f"Warning: no sibling config.json found next to {npz_path}; "
        f"falling back to {args.config}"
    )
    return config_utils.load(args.config, flags=args)


def make_env_config(config, *, num_envs: Optional[int] = None, max_steps=None):
    as_dict = getattr(config.environment, "as_dict", None)
    if callable(as_dict):
        env_config = type(config.environment)(**as_dict())
    else:
        env_config = deepcopy(config.environment)
    if num_envs is not None:
        env_config.common.num_envs = int(num_envs)
        env_config.common.num_threads = int(num_envs)
    if max_steps is not None:
        env_config.common.max_steps = int(max_steps)
    return env_config


def create_raw_env(
    create_env_fn: Callable,
    config,
    split: str,
    level_seeds: Optional[List[int]],
    *,
    num_envs: Optional[int] = None,
    max_steps=None,
):
    env_config = make_env_config(config, num_envs=num_envs, max_steps=max_steps)
    return create_env_fn(
        split,
        env_config,
        level_seeds=level_seeds,
        level_seeds_mode="sequential",
        render_mode=None,
    )


def create_coord_env(
    create_env_fn: Callable,
    config,
    split: str,
    level_seeds: Optional[List[int]],
    weak_agent,
    strong_agent,
    test_eval_info: Dict[str, Any],
):
    base_env = create_raw_env(create_env_fn, config, split, level_seeds)
    coord_env = env_core.CoordEnv(config.coord_env, base_env, weak_agent, strong_agent)
    coord_env.set_costs(test_eval_info)
    coord_env.name = config.environment.common.env_name
    return coord_env


def load_coordination_model_if_needed(policy, config):
    baseline = bool(getattr(config.coord_policy, "baseline", False))
    no_load_algorithms = {
        "timestep_random",
        "level_based_random",
        "oracle_level_based_random",
        "threshold",
        "heuristic",
        "wait",
    }
    if (
        config.general.algorithm != "always"
        and not baseline
        and config.general.algorithm not in no_load_algorithms
    ):
        if config.file_name is None:
            raise ValueError(
                f"Coordination policy for {config.general.algorithm!r} needs "
                "config.file_name to load its model."
            )
        policy.load_model(os.path.join(config.experiment_dir, config.file_name))
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)


def reset_policy_state(policy, num_envs: int) -> None:
    for i in range(num_envs):
        if hasattr(policy, "reset_rolling_average_buffer"):
            policy.reset_rolling_average_buffer(i)
        if hasattr(policy, "reset_episode"):
            policy.reset_episode(i)


def act_coordination_policy(policy, obs, greedy):
    try:
        action, _, _ = policy.act(obs, greedy=greedy, return_scores_and_recons=True)
    except TypeError:
        action = policy.act(obs, greedy=greedy)
    return action


def rollout(policy, env, num_episodes, expected_seeds=None, greedy=False):
    """
    Rollout the policy on a raw environment and collect episode returns.
    Uses the supplied greedy flag for action selection.
    """
    returns = []
    num_completed = 0
    target_episodes = num_episodes
    cumulative_rewards = [0.0] * env.num_envs
    completed_seeds = []
    seed_to_return = {}

    print(f"  Calling reset() - this will start the first {env.num_envs} episodes")
    obs = env.reset()
    print(f"  Environment reset complete. Obs shape: {obs.shape}")

    step_count = 0
    last_debug_step = -1000
    stuck_counter = 0
    last_num_completed = 0
    done = None
    info = {}
    seeds_exhausted = np.array([False] * env.num_envs)

    while num_completed < target_episodes:
        if step_count % 100 == 0:
            print(
                f"  Step {step_count}: Completed {num_completed}/{target_episodes} episodes",
                end="\r",
            )

        if num_completed == last_num_completed:
            stuck_counter += 1
            if stuck_counter > 5000 and step_count > last_debug_step + 1000:
                print(
                    f"\n  DEBUG: No progress for {stuck_counter} steps at {num_completed}/{target_episodes}"
                )
                print(f"  DEBUG: done array = {done}")
                print(
                    f"  DEBUG: cumulative_rewards = {[f'{r:.1f}' for r in cumulative_rewards]}"
                )
                print(
                    "  DEBUG: level_seeds = "
                    f"{get_info_values(info, 'level_seed', env.num_envs, 'N/A')}"
                )
                print(f"  DEBUG: seeds_exhausted = {seeds_exhausted}")
                last_debug_step = step_count
        else:
            stuck_counter = 0
            last_num_completed = num_completed

        action = policy.act(obs, greedy=greedy)
        next_obs, reward, done, info = env.step(action)

        for i in range(env.num_envs):
            cumulative_rewards[i] += reward[i]
            if done[i]:
                level_seed = get_completed_level_seed(info, i)
                if num_completed < target_episodes:
                    returns.append(float(cumulative_rewards[i]))
                    num_completed += 1
                    if level_seed is not None:
                        completed_seeds.append(level_seed)
                        seed_to_return[level_seed] = float(cumulative_rewards[i])

                cumulative_rewards[i] = 0.0

        seeds_exhausted |= get_seeds_exhausted(info, env.num_envs)
        obs = next_obs
        step_count += 1

        if seeds_exhausted.all() and num_completed < target_episodes:
            raise RuntimeError(
                "Sequential level seeds exhausted before rollout completed: "
                f"{num_completed}/{target_episodes} episodes finished."
            )

    print()
    validate_completed_seeds(completed_seeds, expected_seeds)
    return returns, seed_to_return


def validate_completed_seeds(completed_seeds, expected_seeds):
    if expected_seeds is None:
        return

    from collections import Counter

    completed_set = set(completed_seeds)
    expected_set = set(expected_seeds)
    missing_seeds = expected_set - completed_set
    unexpected_seeds = completed_set - expected_set
    seed_counts = Counter(completed_seeds)
    duplicates = {seed: count for seed, count in seed_counts.items() if count > 1}

    print("\n  Validation summary:")
    print(f"    - Expected seeds: {len(expected_seeds)}")
    print(f"    - Completed seeds: {len(completed_set)}")
    print(f"    - Total episodes: {len(completed_seeds)}")
    print(
        "    - All expected seeds completed exactly once: "
        f"{missing_seeds == set() and duplicates == {} and unexpected_seeds == set()}"
    )

    if missing_seeds:
        print(
            f"\n  ERROR: {len(missing_seeds)} expected seeds did not complete: "
            f"{sorted(list(missing_seeds))[:10]}..."
        )
    if duplicates:
        print(f"\n  ERROR: Some seeds completed multiple times: {duplicates}")
    if unexpected_seeds:
        print(
            f"\n  WARNING: {len(unexpected_seeds)} unexpected seeds completed: "
            f"{sorted(list(unexpected_seeds))[:10]}..."
        )
    if missing_seeds or duplicates or unexpected_seeds:
        raise RuntimeError("Seed validation failed during strong re-evaluation.")


def rollout_coordination_sanity(
    policy,
    env,
    record: PointRecord,
    greedy=False,
    defer_to_oracle=False,
):
    update_policy_params(policy, record.threshold)
    policy.eval()

    episode_log = {
        "cumulative_reward": [0.0] * env.num_envs,
        "episode_length": [0] * env.num_envs,
    }
    current_level_ood_gt = [False] * env.num_envs
    current_level_ood_pred = [False] * env.num_envs
    first_help_timestep = [None] * env.num_envs
    pre_help_actions = [[] for _ in range(env.num_envs)]
    completed_pre_help_actions: Dict[int, List[int]] = {}

    obs = env.reset()
    reset_policy_state(policy, env.num_envs)

    completed_seeds = []
    completed_preds = []
    completed_first_help = []
    completed_returns = []
    num_completed = 0
    step_count = 0
    info = {}
    seeds_exhausted = np.array([False] * env.num_envs)

    while num_completed < len(record.level_seeds):
        if step_count % 100 == 0:
            print(
                f"  Sanity step {step_count}: Completed {num_completed}/{len(record.level_seeds)} episodes",
                end="\r",
            )

        obs["episode_timestep"] = episode_log["episode_length"]
        obs["level_ood_gt"] = np.array(current_level_ood_gt, dtype=bool)
        action = act_coordination_policy(policy, obs, greedy)
        original_action = action.copy()

        for i in range(env.num_envs):
            if original_action[i] == env.STRONG:
                current_level_ood_pred[i] = True
            if defer_to_oracle:
                action[i] = int(current_level_ood_pred[i])

        next_obs, reward, done, info = env.step(action)

        for i in range(env.num_envs):
            episode_log["cumulative_reward"][i] += float(reward[i])
            episode_log["episode_length"][i] += 1

            if original_action[i] == env.STRONG and first_help_timestep[i] is None:
                first_help_timestep[i] = episode_log["episode_length"][i]
            elif first_help_timestep[i] is None:
                env_action = get_info_value(info, "env_action", i)
                if env_action is not None:
                    pre_help_actions[i].append(int(env_action))

            if done[i]:
                level_seed = get_completed_level_seed(info, i)
                if level_seed is not None and num_completed < len(record.level_seeds):
                    completed_seeds.append(level_seed)
                    completed_preds.append(current_level_ood_pred[i])
                    completed_first_help.append(first_help_timestep[i])
                    completed_returns.append(float(episode_log["cumulative_reward"][i]))
                    completed_pre_help_actions[level_seed] = list(pre_help_actions[i])
                    num_completed += 1

                episode_log["cumulative_reward"][i] = 0.0
                episode_log["episode_length"][i] = 0
                current_level_ood_pred[i] = False
                first_help_timestep[i] = None
                pre_help_actions[i] = []
                if hasattr(policy, "reset_rolling_average_buffer"):
                    policy.reset_rolling_average_buffer(i)
                if hasattr(policy, "reset_episode"):
                    policy.reset_episode(i)

            current_level_ood_gt[i] = bool(
                get_info_value(info, "randomize_goal", i, False)
            )

        seeds_exhausted |= get_seeds_exhausted(info, env.num_envs)
        if seeds_exhausted.all() and num_completed < len(record.level_seeds):
            raise RuntimeError(
                "Sequential level seeds exhausted before sanity rerun completed: "
                f"{num_completed}/{len(record.level_seeds)} episodes finished."
            )
        obs = next_obs
        step_count += 1

    print()
    return SanityRolloutResult(
        level_seeds=completed_seeds,
        level_ood_pred=completed_preds,
        first_help_timesteps=completed_first_help,
        raw_returns=completed_returns,
        pre_help_actions_by_seed=completed_pre_help_actions,
    )


def validate_sanity_matches(record: PointRecord, sanity: SanityRolloutResult):
    if sanity.level_seeds != record.level_seeds:
        raise AssertionError(
            f"Point {record.index}: level seed order changed.\n"
            f"Original first 10: {record.level_seeds[:10]}\n"
            f"Sanity first 10:   {sanity.level_seeds[:10]}"
        )

    if sanity.level_ood_pred != record.level_ood_pred:
        mismatches = [
            seed
            for seed, expected, actual in zip(
                record.level_seeds, record.level_ood_pred, sanity.level_ood_pred
            )
            if expected != actual
        ]
        raise AssertionError(
            f"Point {record.index}: help decisions changed for "
            f"{len(mismatches)} seeds. First mismatches: {mismatches[:10]}"
        )

    for seed, expected_pred, expected_ts, actual_ts in zip(
        record.level_seeds,
        record.level_ood_pred,
        record.first_help_timesteps,
        sanity.first_help_timesteps,
    ):
        if expected_pred and expected_ts != actual_ts:
            raise AssertionError(
                f"Point {record.index}: first help timestep changed for seed {seed}: "
                f"original={expected_ts}, sanity={actual_ts}"
            )

    if not np.allclose(record.raw_returns, sanity.raw_returns, atol=RETURN_ATOL):
        diffs = np.abs(np.array(record.raw_returns) - np.array(sanity.raw_returns))
        max_idx = int(np.argmax(diffs))
        raise AssertionError(
            f"Point {record.index}: sanity returns differ from original. "
            f"Max diff {diffs[max_idx]:.6g} at seed {record.level_seeds[max_idx]} "
            f"(original={record.raw_returns[max_idx]}, "
            f"sanity={sanity.raw_returns[max_idx]})."
        )


def replay_actions_then_expert(env, strong_policy, pre_help_actions, *, greedy=False):
    obs = env.reset()
    cumulative_reward = 0.0
    done = np.array([False])
    info = [{}]

    for pre_action in pre_help_actions:
        obs, reward, done, info = env.step(np.array([pre_action], dtype=np.int64))
        cumulative_reward += float(reward[0])
        if done[0]:
            raise RuntimeError(
                "Episode ended while replaying pre-help actions before expert takeover."
            )

    while not done[0]:
        action = strong_policy.act(obs, greedy=greedy)
        obs, reward, done, info = env.step(action)
        cumulative_reward += float(reward[0])

    return {
        "return": cumulative_reward,
        "level_seed": get_completed_level_seed(info, 0),
    }


def evaluate_reset_timeout_seed(
    create_env_fn: Callable,
    config,
    split: str,
    seed: int,
    pre_help_actions: List[int],
    strong_policy,
    base_timeout: int,
    greedy: bool,
):
    timeout_cap = reset_timeout_cap(len(pre_help_actions), base_timeout)
    env = create_raw_env(
        create_env_fn,
        config,
        split,
        [seed],
        num_envs=1,
        max_steps=timeout_cap,
    )
    try:
        result = replay_actions_then_expert(
            env, strong_policy, pre_help_actions, greedy=greedy
        )
    finally:
        env.close()

    completed_seed = result["level_seed"]
    if completed_seed is not None and completed_seed != seed:
        raise AssertionError(
            f"Reset-timeout replay completed seed {completed_seed}, expected {seed}"
        )
    result["timeout_cap"] = timeout_cap
    result["pre_help_steps"] = len(pre_help_actions)
    return result


def empty_output(num_points: int):
    return {
        "strong_performances": np.full(num_points, np.nan),
        "original_help_performances": np.full(num_points, np.nan),
        "sanity_performances": np.full(num_points, np.nan),
        "reset_timeout_performances": np.full(num_points, np.nan),
        "comparison_meta": np.array(
            [
                {
                    "point_idx": i,
                    "help_requested": False,
                    "help_seeds": [],
                    "assertions_passed": False,
                    "skip_reason": "no_seeds",
                }
                for i in range(num_points)
            ],
            dtype=object,
        ),
    }


def main():
    args = flags.make()

    if not args.npz_file:
        raise ValueError("Must provide --npz_file argument")
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    strong_agent_path = getattr(args.agents, "strong", None)
    if not strong_agent_path:
        strong_agent_path = getattr(args, "model_file", None)
    if not strong_agent_path:
        raise ValueError("Must provide strong agent path via -strong <path>")

    config = load_reval_config(args, npz_path)
    greedy = resolve_action_greedy(config)
    print(f"Using greedy action selection: {greedy}")

    start_time = time.time()
    use_wandb = os.environ.get("DISABLE_WANDB", "0") != "1"
    wandb_run = None
    job_name = getattr(args, "name", None) or npz_path.stem

    if use_wandb:
        wandb_run = init_eval_wandb_run(
            config,
            name=f"strong_reval_{job_name}",
            job_type="eval_strong",
            run_config={
                "npz_file": args.npz_file,
                "strong_agent": strong_agent_path,
                "config_file": args.config,
                "exp_name": config.exp_name,
                "greedy": greedy,
                "rich_comparison": True,
            },
            project_fallback="yrc-bench-strong-reval",
            tolerate_failure=True,
        )
        if wandb_run is None:
            use_wandb = False

    print(f"\nLoading results from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    afhps = data["afhps"]
    original_performances = data["performances"]
    meta = data["meta"]

    print("\nData summary:")
    print(f"  - Number of points: {len(afhps)}")
    print(f"  - AFHP range: {min(afhps):.2f}% - {max(afhps):.2f}%")
    print(
        f"  - Performance range: {min(original_performances):.2f} - {max(original_performances):.2f}"
    )

    split = "test"
    point_records = [
        extract_point_record(pt_meta, i, split=split) for i, pt_meta in enumerate(meta)
    ]
    all_help_seeds = sorted(
        {seed for record in point_records for seed in help_seeds(record)}
    )
    all_point_seeds = sorted(
        {seed for record in point_records for seed in record.level_seeds}
    )
    print(f"Found {len(all_point_seeds)} unique seeds across all evaluation points")
    print(f"Found {len(all_help_seeds)} unique help-requested seeds")

    if len(all_point_seeds) == 0:
        print("\nWarning: no seeds found in any evaluation point. Saving empty output.")
        output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
        save_npz_results(
            output_path,
            afhps=afhps,
            original_performances=original_performances,
            meta=meta,
            announce=True,
            **empty_output(len(meta)),
        )
        return

    benchmark = config.general.benchmark
    module = importlib.import_module(f"YRC.envs.{benchmark}")
    create_env_fn = getattr(module, "create_env")
    load_policy_fn = getattr(module, "load_policy")

    print(f"\nBenchmark: {benchmark}")
    print(f"Loading weak agent from {config.agents.weak}...")
    print(f"Loading strong agent from {strong_agent_path}...")
    dummy_env = create_raw_env(create_env_fn, config, "train", None)
    weak_agent = load_policy_fn(config.agents.weak, dummy_env)
    strong_agent = load_policy_fn(strong_agent_path, dummy_env)
    dummy_env.close()

    test_eval_info = env_core.get_test_eval_info(config, {})
    policy_env = create_coord_env(
        create_env_fn,
        config,
        split,
        None,
        weak_agent,
        strong_agent,
        test_eval_info,
    )
    try:
        coord_policy = policy_factory.make(config, policy_env)
        load_coordination_model_if_needed(coord_policy, config)
    finally:
        policy_env.close()

    base_timeout = resolve_base_timeout(config)
    print(f"Base timeout for reset-on-help comparison: {base_timeout}")
    defer_to_oracle = bool(getattr(config.evaluation, "defer_to_oracle", False))
    print(f"Same-setup sanity rerun defer_to_oracle: {defer_to_oracle}")

    sanity_results = []
    print("\nRunning per-point same-setup sanity reruns...")
    print("=" * 60)
    for record in point_records:
        print(
            f"\n[{record.index + 1}/{len(point_records)}] Point {record.index} "
            f"(AFHP={afhps[record.index]:.2f}%, threshold={record.threshold:.6g})"
        )

        coord_env = create_coord_env(
            create_env_fn,
            config,
            split,
            record.level_seeds,
            weak_agent,
            strong_agent,
            test_eval_info,
        )
        try:
            sanity = rollout_coordination_sanity(
                coord_policy,
                coord_env,
                record,
                greedy=greedy,
                defer_to_oracle=defer_to_oracle,
            )
        finally:
            coord_env.close()

        validate_sanity_matches(record, sanity)
        sanity_results.append(sanity)
        print("  - Sanity assertions passed")

    strong_seed_to_return = {}
    if all_help_seeds:
        print(f"\nEvaluating expert from beginning on {len(all_help_seeds)} seeds...")
        strong_env = create_raw_env(create_env_fn, config, split, all_help_seeds)
        try:
            _, strong_seed_to_return = rollout(
                strong_agent,
                strong_env,
                len(all_help_seeds),
                expected_seeds=all_help_seeds,
                greedy=greedy,
            )
        finally:
            strong_env.close()

    original_help_performances = []
    sanity_performances = []
    strong_performances = []
    reset_timeout_performances = []
    comparison_meta = []

    print("\nRunning reset-timeout comparisons and building output curves...")
    print("=" * 60)
    for record, sanity in zip(point_records, sanity_results):
        print(
            f"\n[{record.index + 1}/{len(point_records)}] Point {record.index} "
            f"(AFHP={afhps[record.index]:.2f}%, threshold={record.threshold:.6g})"
        )

        reset_seed_results = {}
        for seed in help_seeds(record):
            reset_seed_results[seed] = evaluate_reset_timeout_seed(
                create_env_fn,
                config,
                split,
                seed,
                sanity.pre_help_actions_by_seed[seed],
                strong_agent,
                base_timeout,
                greedy,
            )

        point_comparison = build_point_comparison(
            record,
            sanity.seed_to_return,
            strong_seed_to_return,
            reset_seed_results,
        )
        original_help_performances.append(point_comparison["original_help_performance"])
        sanity_performances.append(point_comparison["sanity_performance"])
        strong_performances.append(point_comparison["strong_performance"])
        reset_timeout_performances.append(point_comparison["reset_timeout_performance"])
        comparison_meta.append(point_comparison["comparison_meta"])

        print(f"  - Help seeds: {len(help_seeds(record))}/{len(record.level_seeds)}")
        print(
            "  - Means: "
            f"original_help={original_help_performances[-1]:.2f}, "
            f"sanity={sanity_performances[-1]:.2f}, "
            f"expert_start={strong_performances[-1]:.2f}, "
            f"reset_timeout={reset_timeout_performances[-1]:.2f}"
        )

        if use_wandb and wandb_run:
            wandb.log(
                {
                    "point_idx": record.index,
                    "point_afhp": float(afhps[record.index]),
                    "num_help_seeds": len(help_seeds(record)),
                    "original_help_performance": original_help_performances[-1],
                    "sanity_performance": sanity_performances[-1],
                    "strong_performance": strong_performances[-1],
                    "reset_timeout_performance": reset_timeout_performances[-1],
                }
            )

    total_time = time.time() - start_time
    valid_strong_perfs = [p for p in strong_performances if not np.isnan(p)]
    print("\n" + "=" * 60)
    print("Re-evaluation completed!")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Unique seeds evaluated: {len(all_point_seeds)}")
    print(f"  - Unique help seeds evaluated by expert: {len(all_help_seeds)}")
    print(f"  - Points processed: {len(meta)}")
    print(f"  - Points with help requested: {len(valid_strong_perfs)}")

    if valid_strong_perfs:
        avg_original_help = float(
            np.nanmean(np.array(original_help_performances, dtype=float))
        )
        avg_sanity = float(np.nanmean(np.array(sanity_performances, dtype=float)))
        avg_strong = float(np.nanmean(np.array(strong_performances, dtype=float)))
        avg_reset = float(np.nanmean(np.array(reset_timeout_performances, dtype=float)))

        print("\nPerformance summary (for points with help):")
        print(f"  - Average original help performance: {avg_original_help:.2f}")
        print(f"  - Average sanity performance: {avg_sanity:.2f}")
        print(f"  - Average expert-from-start performance: {avg_strong:.2f}")
        print(f"  - Average reset-timeout performance: {avg_reset:.2f}")

    output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
    save_npz_results(
        output_path,
        afhps=afhps,
        original_performances=original_performances,
        strong_performances=np.array(strong_performances, dtype=float),
        original_help_performances=np.array(original_help_performances, dtype=float),
        sanity_performances=np.array(sanity_performances, dtype=float),
        reset_timeout_performances=np.array(reset_timeout_performances, dtype=float),
        comparison_meta=np.array(comparison_meta, dtype=object),
        meta=meta,
        announce=True,
    )

    if use_wandb and wandb_run:
        log_data = {
            "total_time": total_time,
            "total_points": len(meta),
            "unique_seeds_evaluated": len(all_point_seeds),
            "unique_help_seeds": len(all_help_seeds),
            "points_with_help": len(valid_strong_perfs),
        }
        if valid_strong_perfs:
            log_data.update(
                {
                    "avg_original_help_performance": avg_original_help,
                    "avg_sanity_performance": avg_sanity,
                    "avg_strong_performance": avg_strong,
                    "avg_reset_timeout_performance": avg_reset,
                }
            )
        wandb.log(log_data)

        artifact = wandb.Artifact(
            f"strong_reval_results_{job_name}",
            type="evaluation_results",
            description=f"Strong agent re-evaluation results for {job_name}",
        )
        artifact.add_file(str(output_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    print(f"Time taken: {total_time:.1f} seconds")
    print(f"Total unique seeds evaluated: {len(all_point_seeds)}")


if __name__ == "__main__":
    main()

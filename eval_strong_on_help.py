from __future__ import annotations

from collections import Counter
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
from YRC.core import Evaluator
from YRC.core.eval_script_utils import init_eval_wandb_run, save_npz_results
from YRC.core.level_seeds import load_level_seed_splits
from YRC.coverage.coverage_search import update_policy_params


@dataclass
class PointRecord:
    index: int
    threshold: float
    split: str
    level_seeds: List[int]
    level_ood_pred: List[bool]
    first_help_timesteps: List[Optional[int]]
    raw_returns: List[float]


def resolve_action_greedy(config):
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
    values = np.asarray(list(values), dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


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


def resolve_afhp_metric(config) -> str:
    metric = getattr(config.evaluation, "threshold_sampler", None)
    if metric == "step_afhp":
        return "step_afhp"
    if metric in {"level_afhp", "ood_percentage", "ood_pred_percentage"}:
        return "level_afhp"
    raise ValueError(f"Unsupported threshold sampler: {metric!r}")


def extract_afhp_from_summary(summary: Dict[str, Any], afhp_metric: str) -> float:
    if afhp_metric == "step_afhp":
        return float(summary["action_1_frac"]) * 100.0
    if afhp_metric == "level_afhp":
        return float(summary["level_afhp"]) * 100.0
    raise ValueError(f"Unsupported AFHP metric: {afhp_metric!r}")


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


def build_strong_reval_point(
    record: PointRecord,
    strong_seed_to_return: Dict[int, float],
) -> Dict[str, Any]:
    indices = help_indices(record)
    seeds = [record.level_seeds[i] for i in indices]
    original_help_returns = [record.raw_returns[i] for i in indices]
    strong_returns = [strong_seed_to_return.get(seed, float("nan")) for seed in seeds]
    return {
        "help_seeds": seeds,
        "original_help_performance": mean_or_nan(original_help_returns),
        "strong_performance": mean_or_nan(strong_returns),
    }


def build_full_budget_point_result(
    record: PointRecord,
    summary: Dict[str, Any],
    afhp_metric: str,
) -> Dict[str, Any]:
    achieved_afhp = extract_afhp_from_summary(summary, afhp_metric)
    performance = float(summary["env_return_mean"])
    return {
        "afhp": achieved_afhp,
        "performance": performance,
        "meta": {
            "point_idx": record.index,
            "threshold": record.threshold,
            "achieved_afhp": achieved_afhp,
            "performance": performance,
            "summary": summary,
        },
    }


def strong_reval_output_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")


def full_budget_output_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_full_budget_eval.npz")


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


def resolve_cli_greedy_override(args) -> bool:
    policy_args = getattr(args, "policy", None)
    greedy = getattr(policy_args, "greedy", None) if policy_args is not None else None
    if greedy is None:
        greedy = getattr(args, "greedy", None)
    return bool(greedy) if greedy is not None else False


def apply_reval_protocol_overrides(config_dict: Dict[str, Any], args) -> None:
    greedy = resolve_cli_greedy_override(args)
    config_dict.setdefault("coord_env", {})["act_greedy"] = greedy
    config_dict.setdefault("policy", {})["greedy"] = greedy


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
        apply_reval_protocol_overrides(config_dict, args)
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


def rollout(policy, env, num_episodes, expected_seeds=None, greedy=False):
    returns = []
    num_completed = 0
    cumulative_rewards = [0.0] * env.num_envs
    completed_seeds = []
    seed_to_return = {}

    obs = env.reset()
    seeds_exhausted = np.array([False] * env.num_envs)

    while num_completed < num_episodes:
        action = policy.act(obs, greedy=greedy)
        next_obs, reward, done, info = env.step(action)

        for i in range(env.num_envs):
            cumulative_rewards[i] += reward[i]
            if done[i]:
                level_seed = get_completed_level_seed(info, i)
                if num_completed < num_episodes:
                    episode_return = float(cumulative_rewards[i])
                    returns.append(episode_return)
                    num_completed += 1
                    if level_seed is not None:
                        completed_seeds.append(level_seed)
                        seed_to_return[level_seed] = episode_return
                cumulative_rewards[i] = 0.0

        seeds_exhausted |= get_seeds_exhausted(info, env.num_envs)
        if seeds_exhausted.all() and num_completed < num_episodes:
            raise RuntimeError(
                "Sequential level seeds exhausted before rollout completed: "
                f"{num_completed}/{num_episodes} episodes finished."
            )
        obs = next_obs

    validate_completed_seeds(completed_seeds, expected_seeds)
    return returns, seed_to_return


def validate_completed_seeds(completed_seeds, expected_seeds):
    if expected_seeds is None:
        return

    completed_set = set(completed_seeds)
    expected_set = set(expected_seeds)
    missing_seeds = expected_set - completed_set
    unexpected_seeds = completed_set - expected_set
    seed_counts = Counter(completed_seeds)
    duplicates = {seed: count for seed, count in seed_counts.items() if count > 1}

    if missing_seeds or duplicates or unexpected_seeds:
        raise RuntimeError(
            "Seed validation failed during evaluation. "
            f"missing={sorted(missing_seeds)[:10]}, "
            f"unexpected={sorted(unexpected_seeds)[:10]}, "
            f"duplicates={duplicates}"
        )


def load_eval_seeds(config) -> List[int]:
    seed_splits = load_level_seed_splits(config, required_splits=("ood_eval",))
    return [int(seed) for seed in seed_splits["ood_eval"]]


def evaluate_full_budget_point(
    record: PointRecord,
    *,
    coord_policy,
    evaluator: Evaluator,
    create_env_fn: Callable,
    config,
    weak_agent,
    strong_agent,
    eval_seeds: List[int],
    test_eval_info: Dict[str, Any],
    base_timeout: int,
    afhp_metric: str,
):
    update_policy_params(coord_policy, record.threshold)
    coord_env = create_coord_env(
        create_env_fn,
        config,
        record.split,
        eval_seeds,
        weak_agent,
        strong_agent,
        test_eval_info,
    )
    coord_env.enable_timeout_reset(base_timeout)
    summary = evaluator.eval(
        coord_policy,
        {record.split: coord_env},
        [record.split],
        num_episodes=len(eval_seeds),
        threshold=record.threshold,
        close_envs=True,
    )[record.split]
    return build_full_budget_point_result(record, summary, afhp_metric)


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
                "full_budget_eval": True,
                "seed_split": "ood_eval",
            },
            project_fallback="yrc-bench-strong-reval",
            tolerate_failure=True,
        )
        if wandb_run is None:
            use_wandb = False

    print(f"\nLoading results from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    afhps = np.asarray(data["afhps"], dtype=float)
    original_performances = np.asarray(data["performances"], dtype=float)
    meta = data["meta"]
    point_records = [
        extract_point_record(pt_meta, i, split="test") for i, pt_meta in enumerate(meta)
    ]

    all_help_seeds = sorted(
        {seed for record in point_records for seed in help_seeds(record)}
    )
    eval_seeds = load_eval_seeds(config)
    base_timeout = resolve_base_timeout(config)
    afhp_metric = resolve_afhp_metric(config)

    print("\nData summary:")
    print(f"  - Number of points: {len(point_records)}")
    print(f"  - Eval seeds: {len(eval_seeds)} from ood_eval")
    print(f"  - Unique help-requested seeds: {len(all_help_seeds)}")
    print(f"  - Full-budget AFHP metric: {afhp_metric}")
    print(f"  - Timeout reset budget: {base_timeout}")

    benchmark = config.general.benchmark
    module = importlib.import_module(f"YRC.envs.{benchmark}")
    create_env_fn = getattr(module, "create_env")
    load_policy_fn = getattr(module, "load_policy")

    dummy_env = create_raw_env(create_env_fn, config, "train", None)
    weak_agent = load_policy_fn(config.agents.weak, dummy_env)
    strong_agent = load_policy_fn(strong_agent_path, dummy_env)
    dummy_env.close()

    test_eval_info = env_core.get_test_eval_info(config, {})
    policy_env = create_coord_env(
        create_env_fn,
        config,
        "test",
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

    strong_seed_to_return = {}
    if all_help_seeds:
        print(f"\nEvaluating expert from beginning on {len(all_help_seeds)} seeds...")
        strong_env = create_raw_env(create_env_fn, config, "test", all_help_seeds)
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
    strong_performances = []
    print("\nBuilding legacy strong reval output...")
    for record in point_records:
        point_result = build_strong_reval_point(record, strong_seed_to_return)
        original_help_performances.append(point_result["original_help_performance"])
        strong_performances.append(point_result["strong_performance"])
        print(
            f"  - Point {record.index}: "
            f"help_seeds={len(point_result['help_seeds'])}, "
            f"original_help={point_result['original_help_performance']:.2f}, "
            f"expert_start={point_result['strong_performance']:.2f}"
        )

    strong_output = strong_reval_output_path(npz_path)
    save_npz_results(
        strong_output,
        afhps=afhps,
        original_performances=original_performances,
        strong_performances=np.array(strong_performances, dtype=float),
        original_help_performances=np.array(original_help_performances, dtype=float),
        meta=meta,
        announce=True,
    )

    evaluator = Evaluator(config, config.environment)
    full_budget_afhps = []
    full_budget_performances = []
    full_budget_meta = []
    print("\nRunning full-budget eval over ood_eval seeds...")
    for record in point_records:
        point_result = evaluate_full_budget_point(
            record,
            coord_policy=coord_policy,
            evaluator=evaluator,
            create_env_fn=create_env_fn,
            config=config,
            weak_agent=weak_agent,
            strong_agent=strong_agent,
            eval_seeds=eval_seeds,
            test_eval_info=test_eval_info,
            base_timeout=base_timeout,
            afhp_metric=afhp_metric,
        )
        full_budget_afhps.append(point_result["afhp"])
        full_budget_performances.append(point_result["performance"])
        full_budget_meta.append(point_result["meta"])
        print(
            f"  - Point {record.index}: "
            f"threshold={record.threshold:.6g}, "
            f"full_budget_afhp={point_result['afhp']:.2f}, "
            f"full_budget_perf={point_result['performance']:.2f}"
        )

    full_budget_output = full_budget_output_path(npz_path)
    save_npz_results(
        full_budget_output,
        thresholds=np.array(
            [record.threshold for record in point_records], dtype=float
        ),
        original_afhps=afhps,
        original_performances=original_performances,
        full_budget_afhps=np.array(full_budget_afhps, dtype=float),
        full_budget_performances=np.array(full_budget_performances, dtype=float),
        full_budget_meta=np.array(full_budget_meta, dtype=object),
        run_metadata=np.array(
            {
                "seed_split": "ood_eval",
                "num_eval_seeds": len(eval_seeds),
                "afhp_metric": afhp_metric,
                "base_timeout": base_timeout,
            },
            dtype=object,
        ),
        announce=True,
    )

    total_time = time.time() - start_time
    avg_original_help = mean_or_nan(original_help_performances)
    avg_strong = mean_or_nan(strong_performances)
    avg_full_budget_afhp = mean_or_nan(full_budget_afhps)
    avg_full_budget_perf = mean_or_nan(full_budget_performances)

    print("\n" + "=" * 60)
    print("Finished strong reval + full-budget eval")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Strong reval output: {strong_output}")
    print(f"  - Full-budget output: {full_budget_output}")
    print(f"  - Average original help performance: {avg_original_help:.2f}")
    print(f"  - Average expert-from-start performance: {avg_strong:.2f}")
    print(f"  - Average full-budget AFHP: {avg_full_budget_afhp:.2f}")
    print(f"  - Average full-budget performance: {avg_full_budget_perf:.2f}")

    if use_wandb and wandb_run:
        wandb.log(
            {
                "total_time": total_time,
                "num_points": len(point_records),
                "num_eval_seeds": len(eval_seeds),
                "num_help_seeds": len(all_help_seeds),
                "avg_original_help_performance": avg_original_help,
                "avg_strong_performance": avg_strong,
                "avg_full_budget_afhp": avg_full_budget_afhp,
                "avg_full_budget_performance": avg_full_budget_perf,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()

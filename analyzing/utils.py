#!/usr/bin/env python3
"""
Utility functions for analyzing environments and agents.
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

import json
import re
from datetime import datetime

import matplotlib

matplotlib.use("TkAgg")


# Add YRC to path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from procgen import ProcgenEnv
from YRC.envs.procgen.wrappers import (
    VecExtractDictObs,
    TransposeFrame,
    ScaledFloatFrame,
    HardResetWrapper,
)


def create_env(random_percent: int = 100, start_level: int = 0, num_levels: int = 1):
    """
    Create a coinrun environment with specified parameters.

    Args:
        random_percent: Percentage of coin randomization (0=deterministic, 100=fully
            random)
        start_level: Starting level seed
        num_levels: Number of levels to include

    Returns:
        Wrapped procgen environment
    """
    # Create base environment
    env = ProcgenEnv(
        env_name="coinrun",
        num_envs=1,
        num_threads=1,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        rand_seed=start_level,  # Use level as seed for consistency
        use_backgrounds=True,
        use_monochrome_assets=False,
        restrict_themes=False,
        random_percent=random_percent,  # Key parameter for counterfactual analysis
    )

    # Apply wrappers (same as in YRC framework)
    env = VecExtractDictObs(env, "rgb")
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    env = HardResetWrapper(env)
    env.obs_shape = env.observation_space.shape

    return env


def get_episode_level_metric(
    test_summary: dict, key: str, success_only: bool = False
) -> np.ndarray:
    """
    Extract episode-level data for a given key from a test summary.

    Args:
        test_summary: The test summary dict from element["summary"]["test"]
        key: The metric key to extract
        success_only: If True, only return values for episodes with reward > 0

    Returns:
        Array of values, one per episode (filtered if success_only=True)
    """
    if success_only:
        raw_returns = np.array(test_summary["raw_returns"])
        success_mask = raw_returns > 0
        for key in test_summary.keys():
            if key == "raw_returns":
                continue
            if isinstance(test_summary[key], np.ndarray):
                test_summary[key] = test_summary[key][success_mask]

    # Get the requested data
    if key == "episode_length" or key == "episode_lengths":
        values = np.array(test_summary["episode_lengths"])
    elif key == "raw_return" or key == "raw_returns":
        values = np.array(test_summary["raw_returns"])
    elif key == "level_ood_gt":
        values = np.array(test_summary["level_ood_gt"])
    elif key == "level_ood_pred":
        values = np.array(test_summary["level_ood_pred"])
    elif key == "first_ood_timestep":
        # Filter out None values for timesteps where OOD was never predicted
        timesteps = test_summary["first_ood_timestep"]
        valid_timesteps = [t for t in timesteps if t is not None]
        values = np.array(valid_timesteps)
    elif key == "ood_prediction_correctness":
        # Whether the OOD prediction matches ground truth (per episode)
        preds = np.array(test_summary["level_ood_pred"])
        gts = np.array(test_summary["level_ood_gt"])
        values = (preds == gts).astype(int)
    else:
        raise ValueError(f"Unknown key: {key}")

    return values


def extract_from_data(data, key: str) -> np.ndarray:
    # Canonicalize these values by turning them into integers.
    level_ood_gt = [
        element["summary"]["test"]["level_ood_gt"] for element in data["meta"]
    ]

    level_ood_pred = [
        element["summary"]["test"]["level_ood_pred"] for element in data["meta"]
    ]
    if key == "ood_pred_percentage":
        pred_percentages = []
        for preds in level_ood_pred:
            percentage = sum(preds) / len(preds)
            pred_percentages.append(percentage)
        return np.array(pred_percentages)
    elif key == "ood_accuracy":
        accs = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            acc = (np.array(preds) == np.array(gts)).mean()
            accs.append(acc)
        return np.array(accs)
    elif key == "true_positive":
        tps = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            tp_count = ((np.array(preds) == 1)[np.array(gts) == 1]).sum()
            pos_count = (np.array(gts) == 1).sum()
            tps.append(tp_count / pos_count)
        return np.array(tps)
    elif key == "false_positive":
        fps = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            fp_count = ((np.array(preds) == 1)[np.array(gts) == 0]).sum()
            neg_count = (np.array(gts) == 0).sum()
            fps.append(fp_count / neg_count)
        return np.array(fps)
    elif key == "true_negative":
        tns = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            tn_count = ((np.array(preds) == 0)[np.array(gts) == 0]).sum()
            neg_count = (np.array(gts) == 0).sum()
            tns.append(tn_count / neg_count)
        return np.array(tns)
    elif key == "false_negative":
        fns = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            fn_count = ((np.array(preds) == 0)[np.array(gts) == 1]).sum()
            pos_count = (np.array(gts) == 1).sum()
            fns.append(fn_count / pos_count)
        return np.array(fns)
    elif key == "performance":
        return data["performances"]
    elif key == "performance_asked":
        # Performance only for episodes where the agent asked for help
        performances = []
        for element in data["meta"]:
            test_summary = element["summary"]["test"]
            raw_returns = np.array(test_summary["raw_returns"])
            level_ood_pred = np.array(test_summary["level_ood_pred"])
            # Filter to episodes where agent asked for help
            asked_mask = level_ood_pred == 1
            if asked_mask.sum() > 0:
                performances.append(raw_returns[asked_mask].mean())
            else:
                performances.append(np.nan)
        return np.array(performances)
    elif key == "performance_not_asked":
        # Performance only for episodes where the agent did NOT ask for help
        performances = []
        for element in data["meta"]:
            test_summary = element["summary"]["test"]
            raw_returns = np.array(test_summary["raw_returns"])
            level_ood_pred = np.array(test_summary["level_ood_pred"])
            # Filter to episodes where agent did not ask for help
            not_asked_mask = level_ood_pred == 0
            if not_asked_mask.sum() > 0:
                performances.append(raw_returns[not_asked_mask].mean())
            else:
                performances.append(np.nan)
        return np.array(performances)
    elif key == "afhp":
        return data["afhps"]
    elif key == "episode_length_mean":
        means = []
        for element in data["meta"]:
            test_summary = element["summary"]["test"]
            episode_lengths = get_episode_level_metric(test_summary, "episode_lengths")
            means.append(np.mean(episode_lengths))
        return np.array(means)
    elif key == "episode_length_success_mean":
        success_means = []
        for element in data["meta"]:
            test_summary = element["summary"]["test"]
            success_lengths = get_episode_level_metric(
                test_summary, "episode_lengths", success_only=True
            )

            if len(success_lengths) > 0:
                success_means.append(np.mean(success_lengths))
            else:
                success_means.append(np.nan)

        return np.array(success_means)
    elif key == "first_ood_timestep_mean":
        means = []
        for element in data["meta"]:
            test_summary = element["summary"]["test"]
            valid_timesteps = get_episode_level_metric(
                test_summary, "first_ood_timestep"
            )
            if len(valid_timesteps) > 0:
                means.append(np.mean(valid_timesteps))
            else:
                means.append(np.nan)
        return np.array(means)
    else:
        raise ValueError(f"Invalid key: {key}")


def filter_duplicate_x_values(
    x: np.ndarray, y: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter out duplicate x-values, keeping only the point with the lowest order value.

    Args:
        x: Array of x-values
        y: Array of y-values
        order: Array of order values that determine which point to keep

    Returns:
        Filtered x and y arrays with duplicate x-values removed
    """
    if len(x) == 0:
        return x, y

    # Find unique x-values and group indices by x-value
    unique_x = np.unique(x)
    keep_indices = []

    for ux in unique_x:
        # Find all indices with this x-value
        indices = np.where(x == ux)[0]
        if len(indices) == 1:
            # No duplicate, keep it
            keep_indices.append(indices[0])
        else:
            # Multiple points with same x-value, keep the one with lowest order
            orders_at_x = order[indices]
            min_order_idx = indices[np.argmin(orders_at_x)]
            keep_indices.append(min_order_idx)

    # Sort indices to maintain original order
    keep_indices = np.sort(keep_indices)

    return x[keep_indices], y[keep_indices]


def extract_x_and_y_values(
    data, x_data_key: str, y_data_key: str
) -> tuple[np.ndarray, np.ndarray]:
    # x = data["afhps"]
    x = extract_from_data(data, x_data_key)
    y = extract_from_data(data, y_data_key)

    # Extract order data for filtering duplicates
    order = data["order"] if "order" in data else np.arange(len(x))

    # Filter out duplicate x-values, keeping only the point with the lowest order
    x, y = filter_duplicate_x_values(x, y, order)

    return x, y


def get_nested_config_value(config: dict, key_path: str):
    """
    Get a value from a nested config dictionary using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., "general.seed")

    Returns:
        The value at the key path, or None if not found
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def parse_timestamp_from_folder(folder_name: str) -> Optional[datetime]:
    """
    Parse timestamp from folder name.

    Expects timestamps in various formats commonly used in folder names:
    - YYYYMMDD_HHMMSS
    - YYYY-MM-DD_HH-MM-SS
    - YYYYMMDDHHMMSS

    Args:
        folder_name: Name of the folder that may contain a timestamp

    Returns:
        datetime object if timestamp found, None otherwise
    """
    # Try different timestamp patterns
    patterns = [
        r"(\d{8}_\d{6})",  # YYYYMMDD_HHMMSS
        r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})",  # YYYY-MM-DD_HH-MM-SS
        r"(\d{14})",  # YYYYMMDDHHMMSS
    ]

    for pattern in patterns:
        match = re.search(pattern, folder_name)
        if match:
            timestamp_str = match.group(1)
            try:
                # Try parsing with different formats
                if "_" in timestamp_str and "-" in timestamp_str:
                    # YYYY-MM-DD_HH-MM-SS format
                    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                elif "_" in timestamp_str:
                    # YYYYMMDD_HHMMSS format
                    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                else:
                    # YYYYMMDDHHMMSS format
                    return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            except ValueError:
                continue

    return None


def select_run_interactive(run_names: List[str], results: dict) -> tuple[str, Path]:
    """
    Interactively select a run from available runs.

    Args:
        run_names: List of available run names
        results: Dictionary mapping run names to data file paths

    Returns:
        Tuple of (selected_run_name, data_path)
    """
    print("\nAvailable runs:")
    for idx, name in enumerate(run_names):
        print(f"  [{idx}] {name}")

    while True:
        try:
            selection = input(f"\nSelect a run (0-{len(run_names) - 1}): ")
            run_idx = int(selection)
            if 0 <= run_idx < len(run_names):
                break
            else:
                print(f"Please enter a number between 0 and {len(run_names) - 1}")
        except ValueError:
            print("Please enter a valid number")

    selected_run = run_names[run_idx]
    data_path = results[selected_run]

    return selected_run, data_path


def extract_results(
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    ablation_key: Optional[List[str]] = None,
    min_timestamp: Optional[str] = None,
) -> dict[str, Path]:
    """
    Extract evaluation results from directory structure.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: List of prefixes to filter runs. Only include runs with any of these
                      prefixes in the name. Multiple prefixes will combine results from all
                      matching runs. If None, include all runs.
        ablation_key: Config key(s) to differentiate multiple runs
        min_timestamp: Minimum timestamp in format YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS.
                      Only include runs with timestamps >= this value.

    Returns:
        Dictionary mapping method names to result file paths
    """
    evals = {}

    # Parse min_timestamp if provided
    min_dt = None
    if min_timestamp is not None:
        min_dt = parse_timestamp_from_folder(min_timestamp)
        if min_dt is None:
            print(
                f"Warning: Could not parse min_timestamp '{min_timestamp}', ignoring filter"
            )

    for child in eval_dir.iterdir():
        # Every child of the eval_dir is a different "grouped run".
        # We want to only select the runs that support the prefix_filter.
        # If multiple prefixes are provided, include runs matching any of them.
        should_include = False
        if child.is_dir():
            if prefix_filter is None:
                should_include = True
            else:
                should_include = any(prefix in child.name for prefix in prefix_filter)

        if should_include:
            for grandchild in child.iterdir():
                # Every grandchild is a different method.
                method_name = grandchild.name
                for run_dir in grandchild.iterdir():
                    # The runs_dirs are different runs with different
                    # timestamps. Potentially, they might have different
                    # hyperparameters.

                    # Filter by timestamp if min_timestamp is provided
                    if min_dt is not None:
                        run_dt = parse_timestamp_from_folder(run_dir.name)
                        if run_dt is None or run_dt < min_dt:
                            continue  # Skip this run

                    for run_file in run_dir.iterdir():
                        if run_file.is_file() and run_file.suffix == ".npz":
                            # If ablation_key is provided, differentiate runs by config value
                            if ablation_key is not None and len(ablation_key) > 0:
                                config_file = run_dir / "config.json"
                                if config_file.exists():
                                    with open(config_file, "r") as f:
                                        config = json.load(f)

                                    # Build unique key from all ablation keys
                                    key_parts = []
                                    for key in ablation_key:
                                        ablation_value = get_nested_config_value(
                                            config, key
                                        )
                                        # Use only the final element of the key path for cleaner labels
                                        key_label = key.split(".")[-1]
                                        key_parts.append(
                                            f"{key_label}={ablation_value}"
                                        )

                                    # Combine method name with all ablation key-value pairs
                                    unique_key = f"{method_name}_{'_'.join(key_parts)}"
                                    evals[unique_key] = run_file
                                else:
                                    # Fallback if config.json doesn't exist
                                    evals[method_name] = run_file
                            else:
                                evals[method_name] = run_file
    return evals

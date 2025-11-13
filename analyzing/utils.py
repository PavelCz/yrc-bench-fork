#!/usr/bin/env python3
"""
Utility functions for analyzing environments and agents.
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore

import argparse
import json
import re
from datetime import datetime


# Add YRC to path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from procgen import ProcgenEnv
from YRC.envs.procgen.wrappers import (
    VecExtractDictObs,
    TransposeFrame,
    ScaledFloatFrame,
    HardResetWrapper,
)
import matplotlib

# The backend must be set before any plotting operations, but after importing
# procgen, which pulls in dependencies that set the backend.
matplotlib.use("TkAgg")


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


def plot_afhp(
    name_order: Optional[List[str]],
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    x_data_key: str,
    y_data_key: str,
    disable_horizontal_lines: bool = False,
    key_filter: Optional[List[str]] = None,
    ablation_key: Optional[List[str]] = None,
    separate_figures: bool = False,
    min_timestamp: Optional[str] = None,
):
    """
    Plot AFHP (Ask for Help Percentage) vs performance.

    Args:
        name_order: List of method names to plot in order. If None, uses all available
        names.
        ablation_key: Config key(s) to differentiate multiple runs from the same method.
        separate_figures: If True, plot each curve in a separate figure in a grid 
        layout.
        min_timestamp: Minimum timestamp to filter runs. Only runs with timestamps >= 
        this value will be included.
    """
    results = extract_results(eval_dir, prefix_filter, ablation_key, min_timestamp)

    # Collect first and last performance values for all curves
    first_performances = []
    last_performances = []

    name_map = {
        "latent-svdd": "Latent SVDD",
        "random": "Timestep Random",
        "patient-ae": "Autoencoder",
        "center-focused": "Center-focused AE",
        "deep-svdd": "DeepSVDD",
        "oc-random": "Level-Based Random",
    }

    # If name_order is None, use all available names
    if name_order is None:
        name_order = list(results.keys())

    if key_filter is not None:
        name_order = [name for name in name_order if name not in key_filter]

    # Filter out methods that don't exist in results
    valid_names = []
    for name in name_order:
        if name not in results:
            print(f"Warning: {name} not found in evals, skipping...")
            continue
        valid_names.append(name)

    if not valid_names:
        print("No valid methods found to plot.")
        return

    if separate_figures:
        # Create a grid layout for separate figures
        n_plots = len(valid_names)
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, name in enumerate(valid_names):
            ax = axes[idx]
            data_path = results[name]

            eval_data = np.load(data_path, allow_pickle=True)
            x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

            # Store first and last performance values
            first_performances.append(y[0])
            last_performances.append(y[-1])

            if name in name_map:
                label = name_map[name]
            else:
                label = name

            sns.lineplot(x=x, y=y, ax=ax, marker="o")
            ax.set_title(f"{label}")
            ax.set_xlabel(x_data_key)
            ax.set_ylabel(y_data_key)

            # Add horizontal lines for this individual plot
            if not disable_horizontal_lines:
                ax.axhline(
                    y=y[0],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Weak Agent",
                )
                ax.axhline(
                    y=y[-1],
                    color="blue",
                    linestyle="--",
                    alpha=0.7,
                    label="Oracle",
                )
                ax.legend()

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()
    else:
        # Original behavior: all curves in one figure
        # Clear previous plot
        plt.clf()

        for name in valid_names:
            data_path = results[name]

            eval_data = np.load(data_path, allow_pickle=True)
            x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

            # Store first and last performance values
            first_performances.append(y[0])
            last_performances.append(y[-1])

            if name in name_map:
                label = name_map[name]
            else:
                label = name

            sns.lineplot(x=x, y=y, label=label, marker="o")

        # Calculate means
        mean_first_performance = np.mean(first_performances)
        mean_last_performance = np.mean(last_performances)

        # Add horizontal lines
        if not disable_horizontal_lines:
            plt.axhline(
                y=mean_first_performance,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Weak Agent",
            )
            plt.axhline(
                y=mean_last_performance,
                color="blue",
                linestyle="--",
                alpha=0.7,
                label="Oracle",
            )

        plt.xlabel(x_data_key)
        plt.ylabel(y_data_key)
        plt.title(f"{y_data_key} over {x_data_key}")
        plt.legend()
        # plt.savefig("afhp_plot.png", dpi=300, bbox_inches="tight")
        plt.show()


def eval_result_plotter():
    parser = argparse.ArgumentParser(
        description="Plot AFHP (Ask for Help Percentage) vs performance"
    )
    parser.add_argument(
        "--name_order",
        "-n",
        default=None,
        type=str,
        help=(
            "Comma-separated list of method names to plot in order. If not specified, "
            "uses all available names.",
        ),
    )  # type: ignore[arg-type]

    parser.add_argument(
        "--eval_dir",
        type=str,
        help="Directory containing the evaluation files.",
    )

    parser.add_argument(
        "--prefix_filter",
        default=None,
        type=str,
        nargs="+",
        help=(
            "Prefix filter(s) for the evaluation files. Can specify one or more "
            "prefixes to combine runs from multiple groups."
        ),
    )

    parser.add_argument(
        "--x_data_key",
        type=str,
        help="Key for the x data.",
    )

    parser.add_argument(
        "--y_data_key",
        type=str,
        help="Key for the y data.",
    )

    parser.add_argument(
        "--disable_horizontal_lines",
        action="store_true",
        help="Disable horizontal lines.",
    )

    parser.add_argument(
        "--key_filter",
        "-f",
        default=None,
        type=str,
        nargs="+",
        help="Filter out these keys.",
    )

    parser.add_argument(
        "--ablation_key",
        default=None,
        type=str,
        nargs="+",
        help=(
            "Config key(s) (with dots for nested keys) to differentiate multiple runs "
            "from the same method. Can specify multiple keys to use all of them. "
            "E.g., 'evaluation.coverage_fraction' or 'general.seed' or both"
        ),
    )

    parser.add_argument(
        "--separate_figures",
        action="store_true",
        help="Show each curve in a separate subplot in a grid layout instead of all in one figure.",
    )

    parser.add_argument(
        "--min_timestamp",
        default=None,
        type=str,
        help=(
            "Minimum timestamp to filter runs. Only runs with timestamps >= this value "
            "will be included. Format: YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS"
        ),
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    x_data_key = args.x_data_key
    y_data_key = args.y_data_key

    # Parse name_order if provided
    name_order = None
    if args.name_order:
        name_order = [name.strip() for name in args.name_order.split(",")]
        print(f"Using specified order: {name_order}")
    else:
        print("Using all available method names")

    # Handle prefix_filter (argparse provides a list if nargs is used)
    prefix_filter = args.prefix_filter
    if prefix_filter:
        print(f"Using prefix filter(s): {prefix_filter}")

    plot_afhp(
        name_order,
        eval_dir,
        prefix_filter,
        x_data_key,
        y_data_key,
        args.disable_horizontal_lines,
        args.key_filter,
        args.ablation_key,
        args.separate_figures,
        args.min_timestamp,
    )


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
    elif key == "afhp":
        return data["afhps"]
    elif key == "episode_length_mean":
        means = []
        for element in data["meta"]:
            episode_lengths = element["summary"]["test"]["episode_lengths"]
            means.append(np.mean(episode_lengths))
        return np.array(means)
    elif key == "episode_length_success_mean":
        success_means = []
        for element in data["meta"]:
            episode_lengths = element["summary"]["test"]["episode_lengths"]
            returns = element["summary"]["test"]["raw_returns"]

            success_lengths = [
                length for length, ret in zip(episode_lengths, returns) if ret > 0
            ]

            if success_lengths:
                success_means.append(np.mean(success_lengths))
            else:
                success_means.append(np.nan)

        return np.array(success_means)
    elif key == "first_ood_timestep_mean":
        means = []
        for element in data["meta"]:
            first_ood_timesteps = element["summary"]["test"]["first_ood_timestep"]
            # Filter out None values (episodes where OOD was never predicted)
            valid_timesteps = [t for t in first_ood_timesteps if t is not None]
            if valid_timesteps:
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
        r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
        r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        r'(\d{14})',  # YYYYMMDDHHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, folder_name)
        if match:
            timestamp_str = match.group(1)
            try:
                # Try parsing with different formats
                if '_' in timestamp_str and '-' in timestamp_str:
                    # YYYY-MM-DD_HH-MM-SS format
                    return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                elif '_' in timestamp_str:
                    # YYYYMMDD_HHMMSS format
                    return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                else:
                    # YYYYMMDDHHMMSS format
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            except ValueError:
                continue
    
    return None


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
            print(f"Warning: Could not parse min_timestamp '{min_timestamp}', ignoring filter")

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

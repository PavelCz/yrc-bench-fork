#!/usr/bin/env python3
"""
Script to plot ICML evaluation results with aggregation across experiments.

Handles directory structure like:
    imcl04_coinrun_exp0/
    imcl04_coinrun_exp1/
    imcl04_maze_exp0/
    ...

Aggregates results for the same method across different experiment IDs,
plotting mean with shaded standard error regions.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from scipy import interpolate

from analyzing.utils import extract_x_and_y_values

matplotlib.use("TkAgg")


# Method display names mapping
METHOD_NAMES = {
    "max_prob": "Max Prob",
    "max_logit": "Max Logit",
    "lb_random": "Level-Based Random",
    "ts_random": "Timestep Random",
    "svdd_image": "Image SVDD",
    "svdd_latent": "Latent SVDD",
    "ensemble": "Ensemble Variance",
    "latent-svdd": "Latent SVDD",
    "random": "Timestep Random",
    "oc-random": "Level-Based Random",
}


def parse_experiment_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse experiment directory name to extract prefix, env, and experiment ID.

    Expected format: {prefix}_{env}_exp{id}
    Examples: imcl04_coinrun_exp0, imcl04_maze_exp1

    Returns:
        Tuple of (prefix, env, exp_id) or None if pattern doesn't match
    """
    # Pattern: prefix_env_expN where env can contain underscores (like maze_afh)
    # But typically: prefix_env_expN
    pattern = r"^(.+)_(coinrun|maze|maze_afh|heist)_exp(\d+)$"
    match = re.match(pattern, dir_name)
    if match:
        prefix = match.group(1)
        env = match.group(2)
        exp_id = int(match.group(3))
        return prefix, env, exp_id
    return None


def parse_method_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse method directory name to extract env, method, and experiment ID.

    Expected format: {env}_{method}_exp{id}
    Examples: coinrun_max_prob_exp0, maze_ensemble_exp1

    Returns:
        Tuple of (env, method, exp_id) or None if pattern doesn't match
    """
    # Pattern: env_method_expN
    pattern = r"^(coinrun|maze|maze_afh|heist)_(.+)_exp(\d+)$"
    match = re.match(pattern, dir_name)
    if match:
        env = match.group(1)
        method = match.group(2)
        exp_id = int(match.group(3))
        return env, method, exp_id
    return None


def extract_icml_results(
    eval_dir: Path,
    prefix_filter: Optional[str] = None,
    env_filter: Optional[str] = None,
) -> Dict[str, Dict[int, Path]]:
    """
    Extract evaluation results from ICML directory structure.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with this prefix
        env_filter: Only include runs for this environment

    Returns:
        Dictionary mapping method names to dict of {exp_id: result_file_path}
    """
    results: Dict[str, Dict[int, Path]] = defaultdict(dict)

    for child in eval_dir.iterdir():
        if not child.is_dir():
            continue

        parsed = parse_experiment_dir(child.name)
        if parsed is None:
            continue

        prefix, env, exp_id = parsed

        # Apply filters
        if prefix_filter is not None and prefix != prefix_filter:
            continue
        if env_filter is not None and env != env_filter:
            continue

        # Find method directories within this experiment
        for method_dir in child.iterdir():
            if not method_dir.is_dir():
                continue

            # Parse method directory name (format: {env}_{method}_exp{id})
            parsed_method = parse_method_dir(method_dir.name)
            if parsed_method is None:
                # Fallback to using the directory name as-is
                method_name = method_dir.name
            else:
                method_env, method_name, method_exp_id = parsed_method
                # Verify consistency with parent directory
                if method_exp_id != exp_id:
                    continue  # Skip mismatched experiment IDs

            # Find the most recent run (by timestamp)
            latest_run = None
            latest_time = None

            for run_dir in method_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # Find .npz file in run directory
                for run_file in run_dir.iterdir():
                    if run_file.is_file() and run_file.suffix == ".npz":
                        # Use directory modification time as proxy for recency
                        mtime = run_dir.stat().st_mtime
                        if latest_time is None or mtime > latest_time:
                            latest_time = mtime
                            latest_run = run_file

            if latest_run is not None:
                results[method_name][exp_id] = latest_run

    return dict(results)


def interpolate_to_common_x(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    num_points: int = 100,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Interpolate multiple curves to a common x-axis for aggregation.

    Args:
        x_arrays: List of x-value arrays
        y_arrays: List of y-value arrays
        num_points: Number of points for interpolation

    Returns:
        Tuple of (common_x, list of interpolated y arrays)
    """
    # Find common x range
    x_min = max(arr.min() for arr in x_arrays)
    x_max = min(arr.max() for arr in x_arrays)

    common_x = np.linspace(x_min, x_max, num_points)
    interpolated_y = []

    for x, y in zip(x_arrays, y_arrays):
        # Sort by x for interpolation
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Use linear interpolation
        f = interpolate.interp1d(
            x_sorted, y_sorted, kind="linear", fill_value="extrapolate"
        )
        interpolated_y.append(f(common_x))

    return common_x, interpolated_y


def plot_icml_results(
    eval_dir: Path,
    prefix_filter: Optional[str],
    env_filter: Optional[str],
    x_data_key: str,
    y_data_key: str,
    method_order: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    use_stderr: bool = True,
    disable_horizontal_lines: bool = False,
    disable_random_line: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot ICML results with aggregation across experiments.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with this prefix
        env_filter: Only include runs for this environment
        x_data_key: Key for x-axis data
        y_data_key: Key for y-axis data
        method_order: Order of methods to plot
        method_filter: Methods to exclude
        use_stderr: If True, use standard error; otherwise use standard deviation
        disable_horizontal_lines: Disable weak/oracle reference lines
        disable_random_line: Disable random baseline diagonal line
        save_path: Path to save the figure
    """
    results = extract_icml_results(eval_dir, prefix_filter, env_filter)

    if not results:
        print("No results found matching the filters.")
        return

    # Determine method order
    if method_order is None:
        method_order = sorted(results.keys())

    if method_filter is not None:
        method_order = [m for m in method_order if m not in method_filter]

    # Filter to valid methods
    valid_methods = [m for m in method_order if m in results]

    if not valid_methods:
        print("No valid methods found to plot.")
        return

    # Set up plot style
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(valid_methods))

    # Store weak/oracle performance for reference lines
    all_first_performances = []
    all_last_performances = []
    # Track x range for random baseline line
    all_x_min = []
    all_x_max = []

    for method_idx, method in enumerate(valid_methods):
        exp_data = results[method]
        exp_ids = sorted(exp_data.keys())

        if len(exp_ids) == 0:
            print(f"Warning: No experiments found for {method}, skipping...")
            continue

        # Load all experiment data
        x_arrays = []
        y_arrays = []

        for exp_id in exp_ids:
            data_path = exp_data[exp_id]
            try:
                eval_data = np.load(data_path, allow_pickle=True)
                x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

                if len(x) > 0:
                    x_arrays.append(x)
                    y_arrays.append(y)

                    # Track first/last for reference lines
                    all_first_performances.append(y[0])
                    all_last_performances.append(y[-1])
                    # Track x range
                    all_x_min.append(x.min())
                    all_x_max.append(x.max())
            except Exception as e:
                print(f"Warning: Failed to load {data_path}: {e}")
                continue

        if len(x_arrays) == 0:
            print(f"Warning: No valid data for {method}, skipping...")
            continue

        # Get display name
        label = METHOD_NAMES.get(method, method)

        if len(x_arrays) == 1:
            # Single experiment, just plot the line
            x, y = x_arrays[0], y_arrays[0]
            sort_idx = np.argsort(x)
            plt.plot(
                x[sort_idx],
                y[sort_idx],
                label=f"{label} (n=1)",
                color=colors[method_idx],
                marker="o",
                markersize=4,
            )
        else:
            # Multiple experiments, aggregate
            common_x, interpolated_y = interpolate_to_common_x(x_arrays, y_arrays)
            y_matrix = np.array(interpolated_y)

            y_mean = np.mean(y_matrix, axis=0)
            y_std = np.std(y_matrix, axis=0)

            if use_stderr:
                y_err = y_std / np.sqrt(len(interpolated_y))
            else:
                y_err = y_std

            n_exps = len(x_arrays)

            # Plot mean line
            plt.plot(
                common_x,
                y_mean,
                label=f"{label} (n={n_exps})",
                color=colors[method_idx],
                linewidth=2,
            )

            # Plot shaded error region
            plt.fill_between(
                common_x,
                y_mean - y_err,
                y_mean + y_err,
                color=colors[method_idx],
                alpha=0.2,
            )

    # Add reference lines
    if not disable_horizontal_lines and all_first_performances:
        mean_first = np.mean(all_first_performances)
        mean_last = np.mean(all_last_performances)

        plt.axhline(
            y=mean_first,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Weak Agent",
        )
        plt.axhline(
            y=mean_last,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label="Oracle",
        )

    # Add random baseline diagonal line (from weak to oracle)
    if not disable_random_line and all_first_performances and all_x_min:
        mean_first = np.mean(all_first_performances)
        mean_last = np.mean(all_last_performances)
        x_min = min(all_x_min)
        x_max = max(all_x_max)

        plt.plot(
            [x_min, x_max],
            [mean_first, mean_last],
            color="gray",
            linestyle=":",
            alpha=0.7,
            linewidth=2,
            label="Random",
        )

    # Labels and title
    env_str = env_filter if env_filter else "all"
    prefix_str = prefix_filter if prefix_filter else "all"
    error_type = "SE" if use_stderr else "SD"

    plt.xlabel(x_data_key)
    plt.ylabel(y_data_key)
    plt.title(f"{y_data_key} vs {x_data_key} ({env_str}, prefix={prefix_str}, shaded={error_type})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def list_available_methods(
    eval_dir: Path, prefix_filter: Optional[str], env_filter: Optional[str]
):
    """List available methods and their experiment coverage."""
    results = extract_icml_results(eval_dir, prefix_filter, env_filter)

    if not results:
        print("No results found matching the filters.")
        return

    print("\nAvailable methods:")
    print("-" * 50)
    for method in sorted(results.keys()):
        exp_ids = sorted(results[method].keys())
        print(f"  {method}: exp{exp_ids}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot ICML evaluation results with aggregation across experiments"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing the evaluation files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix filter for experiment directories (e.g., 'imcl04')",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["coinrun", "maze", "maze_afh", "heist"],
        help="Environment filter",
    )
    parser.add_argument(
        "--x_data_key",
        type=str,
        default="afhp",
        help="Key for the x data (default: afhp)",
    )
    parser.add_argument(
        "--y_data_key",
        type=str,
        default="performance",
        help="Key for the y data (default: performance)",
    )
    parser.add_argument(
        "--method_order",
        "-m",
        type=str,
        default=None,
        help="Comma-separated list of methods to plot in order",
    )
    parser.add_argument(
        "--method_filter",
        "-f",
        type=str,
        nargs="+",
        default=None,
        help="Methods to exclude from plot",
    )
    parser.add_argument(
        "--use_std",
        action="store_true",
        help="Use standard deviation instead of standard error for shaded region",
    )
    parser.add_argument(
        "--disable_horizontal_lines",
        action="store_true",
        help="Disable weak/oracle reference lines",
    )
    parser.add_argument(
        "--disable_random_line",
        action="store_true",
        help="Disable random baseline diagonal line",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the figure (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available methods and exit",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    if args.list:
        list_available_methods(eval_dir, args.prefix, args.env)
        return

    # Parse method order
    method_order = None
    if args.method_order:
        method_order = [m.strip() for m in args.method_order.split(",")]

    plot_icml_results(
        eval_dir=eval_dir,
        prefix_filter=args.prefix,
        env_filter=args.env,
        x_data_key=args.x_data_key,
        y_data_key=args.y_data_key,
        method_order=method_order,
        method_filter=args.method_filter,
        use_stderr=not args.use_std,
        disable_horizontal_lines=args.disable_horizontal_lines,
        disable_random_line=args.disable_random_line,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

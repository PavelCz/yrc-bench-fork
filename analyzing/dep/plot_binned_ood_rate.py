#!/usr/bin/env python3
"""
Interactive script to plot OOD rate (fraction of episodes going OOD at each timestep bin).
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import argparse
from pathlib import Path
import re
from collections import defaultdict
from typing import Union, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from analyzing.utils import extract_x_and_y_values

import matplotlib

matplotlib.use("TkAgg")


def parse_experiment_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse experiment directory name to extract prefix, env, and experiment ID.

    Expected format: {prefix}_{env}_exp{id}
    Examples: imcl04_coinrun_exp0, imcl04_maze_exp1

    Returns:
        Tuple of (prefix, env, exp_id) or None if pattern doesn't match
    """
    # Pattern: prefix_env_expN where env can contain underscores (like maze_afh)
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
    prefix_filter: Optional[List[str]] = None,
    env_filter: Optional[str] = None,
) -> Dict[str, Dict[int, Path]]:
    """
    Extract evaluation results from ICML directory structure.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes (list)
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
        if prefix_filter is not None and prefix not in prefix_filter:
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


def calculate_ood_rate(
    test_summary: dict,
    bins: int,
    success_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate OOD rate based on first OOD timesteps.

    For each bin, calculates the fraction of episodes that had their first OOD
    timestep in that bin, out of all episodes that survived up to that bin.

    Args:
        test_summary: The test summary dict from element["summary"]["test"]
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0
    Returns:
        Tuple of (bin_centers, ood_rates)
    """
    # Filter out episodes where first_ood_timestep is None
    filtered_data = [
        (ts, length, return_value)
        for ts, length, return_value in zip(
            test_summary["first_ood_timestep"],
            test_summary["episode_lengths"],
            test_summary["raw_returns"],
        )
        # if ts is not None
    ]
    if not filtered_data:
        return np.array([]), np.array([])
    first_ood_timesteps, ep_lengths, raw_returns = zip(*filtered_data)

    # Apply success filter if requested
    if success_only:
        success_mask = np.array(raw_returns) > 0
        first_ood_timesteps = np.array(first_ood_timesteps)[success_mask]
        ep_lengths = np.array(ep_lengths)[success_mask]
        raw_returns = np.array(raw_returns)[success_mask]

    # Bin based on first_ood_timesteps
    min_ts = 0
    max_ts = 1000
    bin_edges = np.linspace(min_ts, max_ts, bins + 1)

    ood_rates = []
    x_values = []

    # For each bin, calculate how many episodes had their first OOD timestep in that
    # bin, out of all the episodes that went up to that bin or longer.
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        num_surviving_episodes = 0
        for ep_length, first_ood_ts in zip(ep_lengths, first_ood_timesteps):
            # Filter such that we only count episodes that survived at least past
            # the bin start, but also did not have their first OOD timestep before
            # the bin start.
            if first_ood_ts is None or (
                ep_length >= bin_start and first_ood_ts >= bin_start
            ):
                num_surviving_episodes += 1

        num_first_ood = len(
            [
                first_ood_ts
                for first_ood_ts in first_ood_timesteps
                if first_ood_ts is not None
                and first_ood_ts >= bin_start
                and first_ood_ts < bin_end
            ]
        )

        if num_surviving_episodes > 0:
            ood_rate = num_first_ood / num_surviving_episodes
        else:
            ood_rate = 0.0

        ood_rates.append(ood_rate)
        x_values.append(bin_start)

    return np.array(x_values), np.array(ood_rates)


def plot_barplot_single(
    bin_centers: np.ndarray,
    ood_rates: np.ndarray,
    selected_run: str,
    checkpoint_idx: int,
    level_afhp: float,
    bins: int,
    success_only: bool = False,
):
    """Plot OOD rate as a barplot for a single run."""
    filter_msg = " (success only)" if success_only else ""
    print(f"\nPlotting OOD rate at checkpoint {checkpoint_idx}{filter_msg}")
    print(f"  Level AFHP: {level_afhp:.2f}%")
    print(f"  Number of bins: {bins}")
    print(f"  Mean OOD rate: {np.mean(ood_rates):.2%}")

    # Plot as bar chart
    plt.figure(figsize=(12, 6))
    bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1
    plt.bar(
        bin_centers,
        ood_rates,
        width=bar_width,
        edgecolor="black",
        alpha=0.7,
        linewidth=0.5,
    )

    plt.xlabel("Timestep")
    plt.ylabel("OOD Rate")
    plt.ylim(0, 1.0)

    title_suffix = " (Success Only)" if success_only else ""
    plt.title(
        f"OOD Rate by Timestep{title_suffix}\n"
        f"Run: {selected_run}, Checkpoint: {checkpoint_idx}, "
        f"Level AFHP: {level_afhp:.1f}%"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_barplot_compare(
    all_bin_centers: list,
    all_ood_rates: list,
    all_labels: list,
    success_only: bool,
):
    """Plot OOD rates as barplots for multiple runs."""
    filter_msg = " (success only)" if success_only else ""
    print(
        f"\nPlotting {len(all_ood_rates)} OOD rate curves for comparison{filter_msg}..."
    )

    # Plot all curves
    plt.figure(figsize=(14, 7))

    # Use same color scheme as other plots
    if len(all_ood_rates) <= 10:
        colors = plt.cm.tab10(np.arange(len(all_ood_rates)) / 10)
    elif len(all_ood_rates) <= 20:
        colors = plt.cm.tab20(np.arange(len(all_ood_rates)) / 20)
    else:
        colors = []
        for i in range(len(all_ood_rates)):
            if i < 20:
                colors.append(plt.cm.tab20(i / 20))
            elif i < 40:
                colors.append(plt.cm.tab20b((i - 20) / 20))
            else:
                colors.append(plt.cm.tab20c((i - 40) / 20))

    for idx, (bin_centers, ood_rates, label) in enumerate(
        zip(all_bin_centers, all_ood_rates, all_labels)
    ):
        plt.plot(
            bin_centers,
            ood_rates,
            marker="o",
            label=label,
            color=colors[idx],
            linewidth=2,
            markersize=4,
            alpha=0.8,
        )

    plt.xlabel("Timestep")
    plt.ylabel("OOD Rate")
    plt.ylim(0, 1.0)

    title_suffix = " (Success Only)" if success_only else ""
    plt.title(f"OOD Rate Comparison by Timestep{title_suffix}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_single_run(
    results: dict,
    bins: int,
    success_only: bool,
):
    """Plot OOD rate for a single selected run and checkpoint.

    Args:
        results: Dictionary mapping method names to {exp_id: data_path}
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0
    """
    # Display methods and let user select
    print("\nAvailable methods:")
    method_names = sorted(results.keys())
    for idx, method in enumerate(method_names):
        exp_ids = sorted(results[method].keys())
        print(f"  [{idx}] {method}: experiments {exp_ids}")
    
    while True:
        try:
            selection = input(f"\nSelect a method (0-{len(method_names) - 1}): ")
            method_idx = int(selection)
            if 0 <= method_idx < len(method_names):
                selected_method = method_names[method_idx]
                break
            else:
                print(f"Please enter a number between 0 and {len(method_names) - 1}")
        except ValueError:
            print("Please enter a valid number")
    
    # Select experiment for this method
    exp_data = results[selected_method]
    exp_ids = sorted(exp_data.keys())
    
    if len(exp_ids) == 1:
        selected_exp = exp_ids[0]
        print(f"Using experiment {selected_exp} (only one available)")
    else:
        print(f"\nAvailable experiments for {selected_method}:")
        for idx, exp_id in enumerate(exp_ids):
            print(f"  [{idx}] exp{exp_id}")
        
        while True:
            try:
                selection = input(f"\nSelect an experiment (0-{len(exp_ids) - 1}): ")
                exp_idx = int(selection)
                if 0 <= exp_idx < len(exp_ids):
                    selected_exp = exp_ids[exp_idx]
                    break
                else:
                    print(f"Please enter a number between 0 and {len(exp_ids) - 1}")
            except ValueError:
                print("Please enter a valid number")
    
    data_path = exp_data[selected_exp]
    
    # Load data, select checkpoint, and extract OOD rate
    result = select_and_load_checkpoint_data(
        f"{selected_method}_exp{selected_exp}", data_path, bins, success_only
    )

    if result is None:
        return

    bin_centers, ood_rates, checkpoint_idx, level_afhp = result

    # Plot barplot
    plot_barplot_single(
        bin_centers,
        ood_rates,
        f"{selected_method}_exp{selected_exp}",
        checkpoint_idx,
        level_afhp,
        bins,
        success_only,
    )


def plot_compare_runs(
    results: dict,
    bins: int,
    success_only: bool,
    method_filter: Optional[List[str]] = None,
):
    """Plot OOD rates for selected checkpoints from multiple runs.

    Args:
        results: Dictionary mapping method names to {exp_id: data_path}
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0
        method_filter: Only include these methods
    """
    print("\n=== Multi-Run Comparison Mode ===")
    print("You will select a checkpoint for each method/experiment to compare.\n")

    # Apply method filter
    methods_to_plot = sorted(results.keys())
    if method_filter:
        methods_to_plot = [m for m in methods_to_plot if m in method_filter]
        print(f"Filtering to methods: {', '.join(methods_to_plot)}")

    # Collect data for each method/experiment
    all_bin_centers = []
    all_ood_rates = []
    all_labels = []

    for method in methods_to_plot:
        exp_data = results[method]
        
        # For comparison mode, we could either:
        # 1. Let user select one experiment per method
        # 2. Aggregate all experiments for each method
        # Let's go with option 1 for now
        
        exp_ids = sorted(exp_data.keys())
        
        if len(exp_ids) == 0:
            continue
            
        print(f"\n--- Method: {method} ---")
        
        if len(exp_ids) == 1:
            selected_exp = exp_ids[0]
            print(f"Using experiment {selected_exp} (only one available)")
        else:
            print(f"Available experiments: {exp_ids}")
            print("Select experiment to use for comparison:")
            for idx, exp_id in enumerate(exp_ids):
                print(f"  [{idx}] exp{exp_id}")
            
            while True:
                try:
                    selection = input(f"Select experiment (0-{len(exp_ids) - 1}): ")
                    exp_idx = int(selection)
                    if 0 <= exp_idx < len(exp_ids):
                        selected_exp = exp_ids[exp_idx]
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(exp_ids) - 1}")
                except ValueError:
                    print("Please enter a valid number")
        
        data_path = exp_data[selected_exp]

        # Load data, select checkpoint, and extract OOD rate
        result = select_and_load_checkpoint_data(
            run_name=f"{method}_exp{selected_exp}",
            data_path=data_path,
            bins=bins,
            success_only=success_only,
        )

        if result is None:
            continue

        bin_centers, ood_rates, checkpoint_idx, level_afhp = result

        # Store data and label
        all_bin_centers.append(bin_centers)
        all_ood_rates.append(ood_rates)
        label = f"{method} exp{selected_exp} (Level AFHP: {level_afhp:.1f}%)"
        all_labels.append(label)

        print(f"  Selected checkpoint {checkpoint_idx}")
        print(f"  Mean OOD rate: {np.mean(ood_rates):.2%}")

    if len(all_ood_rates) == 0:
        print("\nNo valid data collected. Exiting.")
        return

    # Plot OOD rates
    plot_barplot_compare(all_bin_centers, all_ood_rates, all_labels, success_only)


def select_and_load_checkpoint_data(
    run_name: str,
    data_path: Path,
    bins: int,
    success_only: bool,
) -> Union[tuple[np.ndarray, np.ndarray, int, float], None]:
    """
    Load data, display checkpoints, get user selection, and calculate OOD rate.

    Args:
        run_name: Name of the run being processed
        data_path: Path to the evaluation data file
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0
    Returns:
        Tuple of (bin_centers, ood_rates, checkpoint_idx, level_afhp) or None if error/no data
    """
    print(f"\nLoading data from: {data_path}")

    # Load the evaluation data
    eval_data = np.load(data_path, allow_pickle=True)

    # Display level_afhp for each checkpoint
    print("\nCheckpoints with level AFHP percentages:")
    level_afhps = []

    for idx, element in enumerate(eval_data["meta"]):
        level_ood_pred = element["summary"]["test"]["level_ood_pred"]
        percentage = sum(level_ood_pred) / len(level_ood_pred) * 100
        level_afhps.append(percentage)

        # Also show AFHP and performance if available
        afhp = eval_data["afhps"][idx] if idx < len(eval_data["afhps"]) else "N/A"
        perf = (
            eval_data["performances"][idx]
            if idx < len(eval_data["performances"])
            else "N/A"
        )

        print(f"  [{idx}] Level AFHP: {percentage:.2f}%, AFHP: {afhp}, Performance: {perf}")

    # Let user select a checkpoint
    while True:
        try:
            selection = input(
                f"\nSelect a checkpoint for '{run_name}' (0-{len(level_afhps) - 1}): "
            )
            checkpoint_idx = int(selection)
            if 0 <= checkpoint_idx < len(level_afhps):
                break
            else:
                print(f"Please enter a number between 0 and {len(level_afhps) - 1}")
        except ValueError:
            print("Please enter a valid number")

    # Extract data for the selected checkpoint
    selected_element = eval_data["meta"][checkpoint_idx]
    test_summary = selected_element["summary"]["test"]

    bin_centers, ood_rates = calculate_ood_rate(
        test_summary=test_summary,
        bins=bins,
        success_only=success_only,
    )

    if len(ood_rates) == 0:
        print(f"No data available for OOD rate at checkpoint {checkpoint_idx}")
        return None

    return bin_centers, ood_rates, checkpoint_idx, level_afhps[checkpoint_idx]


def plot_binned_ood_rate_main():
    """Main function for interactive OOD rate visualization."""
    parser = argparse.ArgumentParser(
        description="Interactive plotter for OOD rate (fraction of episodes going OOD at each timestep)"
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
        nargs="+",
        default=None,
        help="Prefix filter(s) for experiment directories (e.g., 'icml04' or 'icml04 icml05')",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["coinrun", "maze", "maze_afh", "heist"],
        help="Environment filter",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for OOD rate calculation (default: 30)",
    )
    parser.add_argument(
        "--compare_runs",
        action="store_true",
        help="Compare multiple runs by plotting OOD rates for selected checkpoints from each run",
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Only include episodes with reward > 0 in OOD rate calculation",
    )
    parser.add_argument(
        "--method_filter",
        "-m",
        type=str,
        nargs="+",
        default=None,
        help="Only include these methods in comparison mode",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Extract all runs using ICML directory structure
    print("Extracting available runs...")
    results = extract_icml_results(eval_dir, args.prefix, args.env)

    if not results:
        print("No runs found!")
        return

    print(f"\nFound {len(results)} methods:")
    for method in sorted(results.keys()):
        exp_ids = sorted(results[method].keys())
        print(f"  {method}: experiments {exp_ids}")

    if args.compare_runs:
        # Multi-run comparison mode
        plot_compare_runs(
            results=results,
            bins=args.bins,
            success_only=args.success_only,
            method_filter=args.method_filter,
        )
    else:
        # Single run mode
        plot_single_run(
            results=results,
            bins=args.bins,
            success_only=args.success_only,
        )


if __name__ == "__main__":
    plot_binned_ood_rate_main()

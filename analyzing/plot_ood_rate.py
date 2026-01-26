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
    success_only: bool,
) -> list[dict[str, float]]:
    """
    Calculate OOD rate based on first OOD timesteps.

    For each timestep, calculates the fraction of episodes that had their first OOD
    timestep at that timestep, out of all episodes that did not finish before that
    timestep and also did not have their first OOD timestep yet.

    Args:
        test_summary: The test summary dict from element["summary"]["test"]
        success_only: If True, only include episodes with reward > 0
    Returns:
        List of dictionaries, each containing the timestep and the OOD rate.
    """
    # Collect episode data: (first_ood_timestep, episode_length)
    # If first_ood_timestep is None, it means this episode was never marked as OOD
    # We replace these values with positive infinity
    episode_data: list[tuple[float, int]] = []

    for first_ts, length, return_value in zip(
        test_summary["first_ood_timestep"],
        test_summary["episode_lengths"],
        test_summary["raw_returns"],
    ):
        if success_only and return_value <= 0:
            # Optionally, filter out unsuccessful episodes.
            continue
        if first_ts is None:
            episode_data.append((float("inf"), length))
        else:
            episode_data.append((first_ts, length))

    if not episode_data:
        return []
        
    # Debug: Count how many episodes went OOD
    num_ood = sum(1 for first_ts, _ in episode_data if first_ts != float("inf"))
    print(f"    Episodes that went OOD: {num_ood}/{len(episode_data)} ({num_ood/len(episode_data)*100:.1f}%)")

    # Find the range of timesteps to consider (1 to max episode length)
    max_length = max(length for _, length in episode_data)

    ood_rates: list[dict[str, float]] = []

    # For each timestep, calculate the OOD rate
    for current_timestep in range(1, max_length + 1):
        # Count episodes that had their first OOD at this exact timestep
        num_first_ood = sum(
            1 for first_ts, _ in episode_data if first_ts == current_timestep
        )

        # Count surviving episodes: still running and haven't gone OOD yet
        num_surviving_episodes = sum(
            1
            for first_ts, length in episode_data
            if length >= current_timestep and first_ts >= current_timestep
        )

        if num_surviving_episodes == 0:
            # No episodes active at this timestep, skip
            continue

        ood_rates.append(
            {
                "timestep": current_timestep,
                "ood_rate": num_first_ood / num_surviving_episodes,
            }
        )

    return ood_rates


def plot_barplot_compare(
    all_ood_rates: list[list[dict[str, float]]],
    all_labels: list,
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
):
    """Plot OOD rates as barplots for multiple runs.

    Args:
        all_ood_rates: List of OOD rate data for each run
        all_labels: Labels for each run
        success_only: If True, only successful episodes were included
        smooth_window: Window size for running average smoothing (0 = no smoothing)
        num_bins: Number of bins for binned smoothing (0 = no binning)
    """
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

    for idx, (ood_rates, label) in enumerate(zip(all_ood_rates, all_labels)):
        timesteps = [d["timestep"] for d in ood_rates]
        rates = [d["ood_rate"] for d in ood_rates]
        stds = None  # Standard deviation (only computed for binning)

        # Apply binned smoothing if requested
        if num_bins > 0 and len(rates) > 0:
            timesteps_arr = np.array(timesteps)
            rates_arr = np.array(rates)

            # Create bins based on timestep range
            bin_edges = np.linspace(
                timesteps_arr.min(), timesteps_arr.max() + 1, num_bins + 1
            )
            bin_indices = np.digitize(timesteps_arr, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            # Calculate mean rate and std for each bin
            binned_timesteps = []
            binned_rates = []
            binned_stds = []
            for bin_idx in range(num_bins):
                mask = bin_indices == bin_idx
                if np.any(mask):
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    binned_timesteps.append(bin_center)
                    binned_rates.append(np.mean(rates_arr[mask]))
                    binned_stds.append(np.std(rates_arr[mask]))

            timesteps = binned_timesteps
            rates = binned_rates
            stds = binned_stds
        # Apply running average smoothing if requested (only if not binning)
        elif smooth_window > 1 and len(rates) >= smooth_window:
            rates_arr = np.array(rates)
            # Compute running mean
            rates_smooth = np.convolve(
                rates_arr, np.ones(smooth_window) / smooth_window, mode="valid"
            )
            # Compute running std
            stds_list = []
            for i in range(len(rates_smooth)):
                window_data = rates_arr[i : i + smooth_window]
                stds_list.append(np.std(window_data))
            stds = stds_list
            rates = rates_smooth
            # Adjust timesteps to match the smoothed rates (center the window)
            offset = (smooth_window - 1) // 2
            timesteps = timesteps[offset : offset + len(rates)]

        # Plot the line
        plt.plot(
            timesteps,
            rates,
            marker="o",
            label=label,
            color=colors[idx],
            linewidth=2,
            markersize=4,
            alpha=0.8,
        )

        # Plot shaded region for standard deviation if available
        if stds is not None:
            rates_arr = np.array(rates)
            stds_arr = np.array(stds)
            plt.fill_between(
                timesteps,
                rates_arr - stds_arr,
                rates_arr + stds_arr,
                color=colors[idx],
                alpha=0.2,
            )

    plt.xlabel("Timestep")
    plt.ylabel("OOD Rate")
    plt.ylim(bottom=0)

    title_suffix = " (Success Only)" if success_only else ""
    plt.title(f"OOD Rate Comparison by Timestep{title_suffix}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def select_and_load_checkpoint_data(
    run_name: str,
    data_path: Path,
    success_only: bool,
    target_afhp: Optional[float] = None,
) -> Union[tuple[list[dict[str, float]], int, float], None]:
    """
    Load data, display checkpoints, get user selection, and calculate OOD rate.

    Args:
        run_name: Name of the run being processed
        data_path: Path to the evaluation data file
        success_only: If True, only include episodes with reward > 0
        target_afhp: If provided, automatically select checkpoint closest to this AFHP
    Returns:
        Tuple of (ood_rates, checkpoint_idx, ood_percentage) or None if error/no data
    """
    print(f"\nLoading data from: {data_path}")

    # Load the evaluation data
    eval_data = np.load(data_path, allow_pickle=True)

    # Collect checkpoint data
    ood_percentages = []
    afhps = []

    for idx, element in enumerate(eval_data["meta"]):
        level_ood_pred = element["summary"]["test"]["level_ood_pred"]
        percentage = sum(level_ood_pred) / len(level_ood_pred) * 100
        ood_percentages.append(percentage)
        
        # Get AFHP value - prefer to use ood_percentages which are more accurate
        # The afhps array sometimes contains incorrect values
        afhps.append(percentage)  # Use OOD percentage as AFHP

    # Select checkpoint based on target AFHP or user input
    if target_afhp is not None:
        # Find checkpoint closest to target AFHP
        valid_indices = [i for i, afhp in enumerate(afhps) if afhp is not None]
        if not valid_indices:
            print(f"No valid AFHP values found in {run_name}")
            return None
            
        # Find closest AFHP
        closest_idx = min(valid_indices, key=lambda i: abs(afhps[i] - target_afhp))
        checkpoint_idx = closest_idx
        
        actual_afhp = afhps[checkpoint_idx]
        perf = (
            eval_data["performances"][checkpoint_idx]
            if checkpoint_idx < len(eval_data["performances"])
            else "N/A"
        )
        
        print(f"  Selected checkpoint {checkpoint_idx}: AFHP={actual_afhp:.2f}% (target={target_afhp:.2f}%), Performance={perf}")
    else:
        # Manual selection - display all checkpoints
        print("\nCheckpoints with OOD prediction percentages:")
        
        for idx, (ood_pct, afhp) in enumerate(zip(ood_percentages, afhps)):
            perf = (
                eval_data["performances"][idx]
                if idx < len(eval_data["performances"])
                else "N/A"
            )
            
            afhp_str = f"{afhp:.2f}" if afhp is not None else "N/A"
            print(f"  [{idx}] OOD%: {ood_pct:.2f}%, AFHP: {afhp_str}, Performance: {perf}")

        # Let user select a checkpoint
        while True:
            try:
                selection = input(
                    f"\nSelect a checkpoint for '{run_name}' (0-{len(ood_percentages) - 1}): "
                )
                checkpoint_idx = int(selection)
                if 0 <= checkpoint_idx < len(ood_percentages):
                    break
                else:
                    print(f"Please enter a number between 0 and {len(ood_percentages) - 1}")
            except ValueError:
                print("Please enter a valid number")

    # Extract data for the selected checkpoint
    selected_element = eval_data["meta"][checkpoint_idx]
    test_summary = selected_element["summary"]["test"]

    ood_rates: list[dict[str, float]] = calculate_ood_rate(
        test_summary=test_summary,
        success_only=success_only,
    )
    if len(ood_rates) == 0:
        print(f"No data available for OOD rate at checkpoint {checkpoint_idx}")
        return None
    return ood_rates, checkpoint_idx, ood_percentages[checkpoint_idx]


def plot_compare_runs(
    results: dict,
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
    method_filter: Optional[List[str]] = None,
    target_afhp: Optional[float] = None,
):
    """Plot OOD rates for selected checkpoints from multiple runs.

    Args:
        results: Dictionary mapping method names to {exp_id: data_path}
        success_only: If True, only include episodes with reward > 0
        smooth_window: Window size for running average smoothing
        num_bins: Number of bins for binned smoothing
        method_filter: Only include these methods
        target_afhp: If provided, automatically select checkpoints closest to this AFHP
    """
    print("\n=== Multi-Run Comparison Mode ===")
    
    # Apply method filter
    methods_to_plot = sorted(results.keys())
    if method_filter:
        methods_to_plot = [m for m in methods_to_plot if m in method_filter]
        print(f"Filtering to methods: {', '.join(methods_to_plot)}")

    # First, select which experiment to use across all methods
    all_exp_ids = set()
    for method in methods_to_plot:
        all_exp_ids.update(results[method].keys())
    
    exp_ids_list = sorted(all_exp_ids)
    
    if len(exp_ids_list) == 0:
        print("No experiments found!")
        return
    elif len(exp_ids_list) == 1:
        selected_exp = exp_ids_list[0]
        print(f"\nUsing experiment {selected_exp} (only one available)")
    else:
        print(f"\nAvailable experiments: {exp_ids_list}")
        print("Select one experiment to use for ALL methods:")
        for idx, exp_id in enumerate(exp_ids_list):
            print(f"  [{idx}] exp{exp_id}")
        
        while True:
            try:
                selection = input(f"Select experiment (0-{len(exp_ids_list) - 1}): ")
                exp_idx = int(selection)
                if 0 <= exp_idx < len(exp_ids_list):
                    selected_exp = exp_ids_list[exp_idx]
                    break
                else:
                    print(f"Please enter a number between 0 and {len(exp_ids_list) - 1}")
            except ValueError:
                print("Please enter a valid number")
    
    print(f"\nUsing experiment {selected_exp} for all methods")
    
    # Get target AFHP from user if not provided
    if target_afhp is None:
        while True:
            try:
                afhp_input = input("\nEnter target AFHP percentage (e.g., 10.5 for 10.5%): ")
                target_afhp = float(afhp_input)
                if 0 <= target_afhp <= 100:
                    break
                else:
                    print("AFHP must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
    
    print(f"\nSelecting checkpoints closest to AFHP={target_afhp:.2f}% for each method...")

    # Collect data for each method
    all_ood_rates = []
    all_labels = []
    
    for method in methods_to_plot:
        exp_data = results[method]
        
        # Check if this method has the selected experiment
        if selected_exp not in exp_data:
            print(f"\n--- Method: {method} ---")
            print(f"  WARNING: No data for experiment {selected_exp}, skipping...")
            continue
            
        print(f"\n--- Method: {method} ---")
        data_path = exp_data[selected_exp]

        # Load data and automatically select checkpoint based on target AFHP
        result = select_and_load_checkpoint_data(
            run_name=f"{method}_exp{selected_exp}",
            data_path=data_path,
            success_only=success_only,
            target_afhp=target_afhp,
        )

        if result is None:
            continue

        ood_rates, checkpoint_idx, ood_percentage = result

        # Store data and label
        all_ood_rates.append(ood_rates)
        label = f"{method} (OOD: {ood_percentage:.1f}%)"
        all_labels.append(label)

        rates = [d["ood_rate"] for d in ood_rates]
        mean_rate = np.mean(rates) if rates else 0.0
        print(f"  Mean OOD rate: {mean_rate:.2%}")

    if len(all_ood_rates) == 0:
        print("\nNo valid data collected. Exiting.")
        return

    # Plot OOD rates
    plot_barplot_compare(all_ood_rates, all_labels, success_only, smooth_window, num_bins)


def plot_single_run(
    results: dict,
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
    target_afhp: Optional[float] = None,
):
    """Plot OOD rate for a single selected run and checkpoint.

    Args:
        results: Dictionary mapping method names to {exp_id: data_path}
        success_only: If True, only include episodes with reward > 0
        smooth_window: Window size for running average smoothing
        num_bins: Number of bins for binned smoothing
        target_afhp: If provided, automatically select checkpoint closest to this AFHP
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
        f"{selected_method}_exp{selected_exp}", data_path, success_only, target_afhp
    )

    if result is None:
        return

    ood_rates, checkpoint_idx, ood_percentage = result

    # Plot single run
    all_ood_rates = [ood_rates]
    all_labels = [f"{selected_method}_exp{selected_exp} (OOD: {ood_percentage:.1f}%)"]
    
    plot_barplot_compare(all_ood_rates, all_labels, success_only, smooth_window, num_bins)


def plot_ood_rate_main():
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
        "--success_only",
        action="store_true",
        help="Only include episodes with reward > 0 in OOD rate calculation",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Window size for running average smoothing (default: 0, no smoothing)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=0,
        help="Number of bins for binned smoothing (default: 0, no binning). Overrides --smooth.",
    )
    parser.add_argument(
        "--compare_runs",
        action="store_true",
        help="Compare multiple runs by plotting OOD rates for selected checkpoints from each run",
    )
    parser.add_argument(
        "--method_filter",
        "-m",
        type=str,
        nargs="+",
        default=None,
        help="Only include these methods in comparison mode",
    )
    parser.add_argument(
        "--target_afhp",
        type=float,
        default=None,
        help="Target AFHP percentage to automatically select closest checkpoints (e.g., 10.5 for 10.5%)",
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
            success_only=args.success_only,
            smooth_window=args.smooth,
            num_bins=args.bins,
            method_filter=args.method_filter,
            target_afhp=args.target_afhp,
        )
    else:
        # Single run mode
        plot_single_run(
            results=results,
            success_only=args.success_only,
            smooth_window=args.smooth,
            num_bins=args.bins,
            target_afhp=args.target_afhp,
        )


if __name__ == "__main__":
    plot_ood_rate_main()

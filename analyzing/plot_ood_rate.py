#!/usr/bin/env python3
"""
Interactive script to plot OOD rate (fraction of episodes going OOD at each timestep bin).
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from analyzing.utils import extract_results, select_run_interactive

import matplotlib

matplotlib.use("TkAgg")


def calculate_ood_rate(
    test_summary: dict, bins: int = 30, success_only: bool = False
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
        if ts is not None
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
    max_ts = max(first_ood_timesteps)
    bin_edges = np.linspace(min_ts, max_ts, bins + 1)

    ood_rates = []
    x_values = []

    # For each bin, calculate how many episodes had their first OOD timestep in that
    # bin, out of all the episodes that went up to that bin or longer.
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        num_surviving_episodes = len(
            [ep_length for ep_length in ep_lengths if ep_length >= bin_start]
        )
        num_first_ood = len(
            [
                first_ood_ts
                for first_ood_ts in first_ood_timesteps
                if first_ood_ts >= bin_start and first_ood_ts < bin_end
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
    ood_percentage: float,
    bins: int,
    success_only: bool = False,
):
    """Plot OOD rate as a barplot for a single run."""
    filter_msg = " (success only)" if success_only else ""
    print(f"\nPlotting OOD rate at checkpoint {checkpoint_idx}{filter_msg}")
    print(f"  OOD prediction percentage: {ood_percentage:.2f}%")
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
        f"OOD%: {ood_percentage:.1f}%"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_barplot_compare(
    all_bin_centers: list,
    all_ood_rates: list,
    all_labels: list,
    success_only: bool = False,
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
    run_names: list,
    results: dict,
    bins: int = 30,
    success_only: bool = False,
):
    """Plot OOD rate for a single selected run and checkpoint.

    Args:
        run_names: List of available run names
        results: Dictionary mapping run names to data file paths
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0
    """
    # Display runs and let user select
    selected_run, data_path = select_run_interactive(run_names, results)

    # Load data, select checkpoint, and extract OOD rate
    result = select_and_load_checkpoint_data(
        selected_run, data_path, bins, success_only
    )

    if result is None:
        return

    bin_centers, ood_rates, checkpoint_idx, ood_percentage = result

    # Plot barplot
    plot_barplot_single(
        bin_centers,
        ood_rates,
        selected_run,
        checkpoint_idx,
        ood_percentage,
        bins,
        success_only,
    )


def plot_compare_runs(
    run_names: list,
    results: dict,
    bins: int = 30,
    success_only: bool = False,
):
    """Plot OOD rates for selected checkpoints from multiple runs.

    Args:
        run_names: List of available run names
        results: Dictionary mapping run names to data file paths
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0
    """
    print("\n=== Multi-Run Comparison Mode ===")
    print("You will select a checkpoint for each run to compare.\n")

    # Collect data for each run
    all_bin_centers = []
    all_ood_rates = []
    all_labels = []

    for run_name in run_names:
        data_path = results[run_name]
        print(f"\n--- Run: {run_name} ---")

        # Load data, select checkpoint, and extract OOD rate
        result = select_and_load_checkpoint_data(
            run_name, data_path, bins, success_only
        )

        if result is None:
            continue

        bin_centers, ood_rates, checkpoint_idx, ood_percentage = result

        # Store data and label
        all_bin_centers.append(bin_centers)
        all_ood_rates.append(ood_rates)
        label = f"{run_name} (OOD: {ood_percentage:.1f}%)"
        all_labels.append(label)

        print(f"  Selected checkpoint {checkpoint_idx}")
        print(f"  Mean OOD rate: {np.mean(ood_rates):.2%}")

    if len(all_ood_rates) == 0:
        print("\nNo valid data collected. Exiting.")
        return

    # Plot OOD rates
    plot_barplot_compare(all_bin_centers, all_ood_rates, all_labels, success_only)


def select_and_load_checkpoint_data(
    run_name: str, data_path: Path, bins: int, success_only: bool = False
) -> Union[tuple[np.ndarray, np.ndarray, int, float], None]:
    """
    Load data, display checkpoints, get user selection, and calculate OOD rate.

    Args:
        run_name: Name of the run being processed
        data_path: Path to the evaluation data file
        bins: Number of bins for OOD rate calculation
        success_only: If True, only include episodes with reward > 0

    Returns:
        Tuple of (bin_centers, ood_rates, checkpoint_idx, ood_percentage) or None if error/no data
    """
    print(f"\nLoading data from: {data_path}")

    # Load the evaluation data
    eval_data = np.load(data_path, allow_pickle=True)

    # Display ood_pred_percentage for each checkpoint
    print("\nCheckpoints with OOD prediction percentages:")
    ood_percentages = []

    for idx, element in enumerate(eval_data["meta"]):
        level_ood_pred = element["summary"]["test"]["level_ood_pred"]
        percentage = sum(level_ood_pred) / len(level_ood_pred) * 100
        ood_percentages.append(percentage)

        # Also show AFHP and performance if available
        afhp = eval_data["afhps"][idx] if idx < len(eval_data["afhps"]) else "N/A"
        perf = (
            eval_data["performances"][idx]
            if idx < len(eval_data["performances"])
            else "N/A"
        )

        print(f"  [{idx}] OOD%: {percentage:.2f}%, AFHP: {afhp}, Performance: {perf}")

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

    bin_centers, ood_rates = calculate_ood_rate(
        test_summary, bins, success_only
    )

    if len(ood_rates) == 0:
        print(f"No data available for OOD rate at checkpoint {checkpoint_idx}")
        return None

    return bin_centers, ood_rates, checkpoint_idx, ood_percentages[checkpoint_idx]


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
        "--prefix_filter",
        default=None,
        type=str,
        help="Prefix filter for the evaluation files.",
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

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Extract all runs
    print("Extracting available runs...")
    results = extract_results(eval_dir, [args.prefix_filter])

    if not results:
        print("No runs found!")
        return

    run_names = list(results.keys())

    if args.compare_runs:
        # Multi-run comparison mode
        plot_compare_runs(run_names, results, args.bins, args.success_only)
    else:
        # Single run mode
        plot_single_run(run_names, results, args.bins, args.success_only)


if __name__ == "__main__":
    plot_ood_rate_main()

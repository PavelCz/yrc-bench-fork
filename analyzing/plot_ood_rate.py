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

from analyzing.utils import extract_results

import matplotlib

matplotlib.use("TkAgg")


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
            1 for first_ts, length in episode_data
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
) -> Union[tuple[list[dict[str, float]], int, float], None]:
    """
    Load data, display checkpoints, get user selection, and calculate OOD rate.

    Args:
        run_name: Name of the run being processed
        data_path: Path to the evaluation data file
        success_only: If True, only include episodes with reward > 0
    Returns:
        Tuple of (ood_rates, checkpoint_idx, ood_percentage) or None if error/no data
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

    ood_rates: list[dict[str, float]] = calculate_ood_rate(
        test_summary=test_summary,
        success_only=success_only,
    )
    if len(ood_rates) == 0:
        print(f"No data available for OOD rate at checkpoint {checkpoint_idx}")
        return None
    return ood_rates, checkpoint_idx, ood_percentages[checkpoint_idx]


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

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Extract all runs
    print("Extracting available runs...")
    results = extract_results(eval_dir, [args.prefix_filter])

    if not results:
        print("No runs found!")
        return

    run_names = list(results.keys())

    # Multi-run comparison mode
    print("\n=== Multi-Run Comparison Mode ===")
    print("You will select a checkpoint for each run to compare.\n")

    # Collect data for each run
    all_ood_rates = []
    all_labels = []

    for run_name in run_names:
        data_path = results[run_name]
        print(f"\n--- Run: {run_name} ---")

        # Load data, select checkpoint, and extract OOD rate
        result = select_and_load_checkpoint_data(
            run_name=run_name,
            data_path=data_path,
            success_only=args.success_only,
        )

        if result is None:
            continue

        ood_rates, checkpoint_idx, ood_percentage = result

        # Store data and label
        all_ood_rates.append(ood_rates)
        label = f"{run_name} (OOD: {ood_percentage:.1f}%)"
        all_labels.append(label)

        rates = [d["ood_rate"] for d in ood_rates]
        mean_rate = np.mean(rates) if rates else 0.0
        print(f"  Selected checkpoint {checkpoint_idx}")
        print(f"  Mean OOD rate: {mean_rate:.2%}")

    if len(all_ood_rates) == 0:
        print("\nNo valid data collected. Exiting.")
        return

    # Plot OOD rates
    plot_barplot_compare(
        all_ood_rates, all_labels, args.success_only, args.smooth, args.bins
    )


if __name__ == "__main__":
    plot_ood_rate_main()

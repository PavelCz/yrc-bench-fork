#!/usr/bin/env python3
"""
Interactive script to plot OOD rate (fraction of episodes going OOD at each timestep bin).
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Union, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("TkAgg")

# Import shared plotting configuration
from analyzing.plotting_common import (
    setup_plot_style,
    get_line_styles,
    style_plot_for_publication,
)

# Reuse robust-variant parsing, canonicalization, and label/legend helpers
# from icml_plot so this script behaves the same way around robust strong
# policies and kebab/snake method aliases.
from analyzing.icml_plot import (
    SUPPORTED_ENVS,
    add_robust_suffix,
    canonicalize_method,
    format_plot_label,
    method_is_filtered,
    method_is_included,
    parse_experiment_dir,
    parse_method_dir,
    parse_robust_experiment_dir,
    prefix_matches_filter,
)


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
    # Match icml_plot behavior: when multiple prefixes provide the same
    # canonical method and experiment id, keep the newest timestamped run.
    latest_run_dir_name: Dict[Tuple[str, int], str] = {}

    for child in sorted(eval_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        # Try the robust-variant pattern ({prefix}_robust{200,400}_{env}_exp{id})
        # first so the variant gets pulled out cleanly; fall back to the plain
        # {prefix}_{env}_exp{id} pattern otherwise.
        robust_parsed = parse_robust_experiment_dir(child.name)
        robust_variant: Optional[str] = None
        if robust_parsed is not None:
            prefix, robust_variant, env, exp_id = robust_parsed
        else:
            parsed = parse_experiment_dir(child.name)
            if parsed is None:
                continue
            prefix, env, exp_id = parsed

        # `prefix_matches_filter` accepts both base prefixes and full
        # `{prefix}_{robust_variant}` prefixes so the user can pass either.
        if not prefix_matches_filter(prefix, robust_variant, prefix_filter):
            continue
        if env_filter is not None and env != env_filter:
            continue

        for method_dir in sorted(child.iterdir(), key=lambda p: p.name):
            if not method_dir.is_dir():
                continue

            parsed_method = parse_method_dir(method_dir.name)
            if parsed_method is None:
                method_name = method_dir.name
            else:
                method_env, method_name, method_exp_id = parsed_method
                if method_exp_id != exp_id:
                    continue

            # Tag with robust variant so robust vs non-robust runs at the same
            # method don't collide on the same method key, then canonicalize.
            method_name = add_robust_suffix(method_name, robust_variant)
            method_name = canonicalize_method(method_name)

            # Pick the run-dir with the largest timestamp name.
            latest_run = None
            latest_name = None
            for run_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
                if not run_dir.is_dir():
                    continue
                for run_file in run_dir.iterdir():
                    if run_file.is_file() and run_file.suffix == ".npz":
                        if latest_name is None or run_dir.name > latest_name:
                            latest_name = run_dir.name
                            latest_run = run_file
                        break

            if latest_run is None:
                continue

            key = (method_name, exp_id)
            prev_name = latest_run_dir_name.get(key)
            if prev_name is not None and latest_name <= prev_name:
                continue
            latest_run_dir_name[key] = latest_name
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
    print(
        f"    Episodes that went OOD: {num_ood}/{len(episode_data)} ({num_ood / len(episode_data) * 100:.1f}%)"
    )

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


def calculate_minmax_bands_ood(
    ood_rates_by_exp: List[List[dict]],
    quantiles: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate min-max quantile bands from multiple OOD rate curves.

    Since OOD rates are calculated at standardized timesteps (0-1000 or 0-500),
    we can directly aggregate without interpolation.

    Args:
        ood_rates_by_exp: List of OOD rate data for each experiment
        quantiles: Tuple of (lower_quantile, upper_quantile), default (0.0, 1.0) for min/max

    Returns:
        Tuple of (common_timesteps, median_rates, lower_quantile_rates, upper_quantile_rates)
    """
    # Collect all timesteps to get the full range
    all_timesteps = set()

    # Build a mapping from timestep to rates across experiments
    timestep_to_rates = defaultdict(list)

    for ood_rates in ood_rates_by_exp:
        for data_point in ood_rates:
            ts = data_point["timestep"]
            rate = data_point["ood_rate"]
            timestep_to_rates[ts].append(rate)
            all_timesteps.add(ts)

    # Sort timesteps
    common_timesteps = np.array(sorted(all_timesteps))

    # Calculate statistics at each timestep
    medians = []
    lower_quantiles = []
    upper_quantiles = []

    for ts in common_timesteps:
        rates_at_ts = np.array(timestep_to_rates[ts])

        if len(rates_at_ts) > 0:
            median = np.median(rates_at_ts)
            lower_q = np.quantile(rates_at_ts, quantiles[0])
            upper_q = np.quantile(rates_at_ts, quantiles[1])

            medians.append(median)
            lower_quantiles.append(lower_q)
            upper_quantiles.append(upper_q)
        else:
            # This shouldn't happen if all experiments have same timesteps
            medians.append(np.nan)
            lower_quantiles.append(np.nan)
            upper_quantiles.append(np.nan)

    return (
        common_timesteps,
        np.array(medians),
        np.array(lower_quantiles),
        np.array(upper_quantiles),
    )


def plot_barplot_compare_aggregated(
    all_method_ood_rates: List[List[List[dict]]],
    all_labels: List[str],
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
    save_path: Optional[str] = None,
    paper_mode: bool = False,
    max_ood_rate: Optional[float] = None,
    max_timesteps: Optional[int] = None,
):
    """Plot OOD rates with aggregation across experiments.

    Args:
        all_method_ood_rates: For each method, list of OOD rate data from each experiment
        all_labels: Labels for each method
        success_only: If True, only successful episodes were included
        smooth_window: Window size for running average smoothing (0 = no smoothing)
        num_bins: Number of bins for binned smoothing (0 = no binning)
        save_path: If provided, save figure to this path instead of displaying
        paper_mode: If True, use paper-ready styling
        max_ood_rate: If provided, set y-axis maximum to this value
        max_timesteps: If provided, only plot up to this many timesteps
    """
    filter_msg = " (success only)" if success_only else ""
    print(
        f"\nPlotting {len(all_method_ood_rates)} methods with aggregation{filter_msg}..."
    )

    # Set up plot style
    setup_plot_style(paper_mode=paper_mode, use_latex=True)

    plt.figure(figsize=(8, 4.5))

    # Use same color scheme as other plots
    colors = []
    n_methods = len(all_method_ood_rates)
    if n_methods <= 10:
        colors = plt.cm.tab10(np.arange(n_methods) / 10)
    elif n_methods <= 20:
        colors = plt.cm.tab20(np.arange(n_methods) / 20)
    else:
        for i in range(n_methods):
            if i < 20:
                colors.append(plt.cm.tab20(i / 20))
            elif i < 40:
                colors.append(plt.cm.tab20b((i - 20) / 20))
            else:
                colors.append(plt.cm.tab20c((i - 40) / 20))

    # Get line styles for paper mode
    line_styles = get_line_styles(n_methods, paper_mode)

    for method_idx, (method_ood_rates_by_exp, label) in enumerate(
        zip(all_method_ood_rates, all_labels)
    ):
        # Calculate aggregated statistics using min/max bands
        common_timesteps, median_rates, min_rates, max_rates = (
            calculate_minmax_bands_ood(method_ood_rates_by_exp, quantiles=(0.0, 1.0))
        )

        # Filter out NaN values
        valid_mask = ~np.isnan(median_rates)
        timesteps = common_timesteps[valid_mask]
        median_rates = median_rates[valid_mask]
        min_rates = min_rates[valid_mask]
        max_rates = max_rates[valid_mask]

        # Apply max timesteps filter if specified
        if max_timesteps is not None and len(timesteps) > 0:
            timestep_mask = timesteps <= max_timesteps
            timesteps = timesteps[timestep_mask]
            median_rates = median_rates[timestep_mask]
            min_rates = min_rates[timestep_mask]
            max_rates = max_rates[timestep_mask]

        # Apply binned smoothing if requested
        if num_bins > 0 and len(timesteps) > 0:
            # Create bins based on timestep range
            bin_edges = np.linspace(timesteps.min(), timesteps.max() + 1, num_bins + 1)
            bin_indices = np.digitize(timesteps, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            # Calculate statistics for each bin
            binned_timesteps = []
            binned_medians = []
            binned_mins = []
            binned_maxs = []

            for bin_idx in range(num_bins):
                mask = bin_indices == bin_idx
                if np.any(mask):
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    binned_timesteps.append(bin_center)
                    binned_medians.append(np.mean(median_rates[mask]))
                    binned_mins.append(np.mean(min_rates[mask]))
                    binned_maxs.append(np.mean(max_rates[mask]))

            timesteps = np.array(binned_timesteps)
            median_rates = np.array(binned_medians)
            min_rates = np.array(binned_mins)
            max_rates = np.array(binned_maxs)

        # Apply running average smoothing if requested (only if not binning)
        elif smooth_window > 1 and len(timesteps) >= smooth_window:
            # Smooth each curve separately
            median_smooth = np.convolve(
                median_rates, np.ones(smooth_window) / smooth_window, mode="valid"
            )
            min_smooth = np.convolve(
                min_rates, np.ones(smooth_window) / smooth_window, mode="valid"
            )
            max_smooth = np.convolve(
                max_rates, np.ones(smooth_window) / smooth_window, mode="valid"
            )

            # Adjust timesteps to match the smoothed rates
            offset = (smooth_window - 1) // 2
            timesteps = timesteps[offset : offset + len(median_smooth)]
            median_rates = median_smooth
            min_rates = min_smooth
            max_rates = max_smooth

        # Plot the median line
        plt.plot(
            timesteps,
            median_rates,
            label=label,
            color=colors[method_idx],
            linewidth=2,
            linestyle=line_styles[method_idx],
            alpha=0.8,
        )

        # Plot shaded region for min/max
        plt.fill_between(
            timesteps,
            min_rates,
            max_rates,
            color=colors[method_idx],
            alpha=0.2,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Ask For Help Probability")

    # Set axis limits
    if max_ood_rate is not None:
        plt.ylim(bottom=0, top=max_ood_rate)
    else:
        plt.ylim(bottom=0)

    if max_timesteps is not None:
        plt.xlim(left=0, right=max_timesteps)

    # Title only if not in paper mode
    if not paper_mode:
        title_suffix = " (Success Only)" if success_only else ""
        plt.title(f"OOD Rate Comparison by Timestep (Aggregated){title_suffix}")

    # Apply publication styling
    style_plot_for_publication(
        legend_outside=True,
        legend_location="center left",
        legend_bbox_to_anchor=(1.05, 0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved figure to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_barplot_compare(
    all_ood_rates: list[list[dict[str, float]]],
    all_labels: list,
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
    save_path: Optional[str] = None,
    paper_mode: bool = False,
    max_ood_rate: Optional[float] = None,
    max_timesteps: Optional[int] = None,
):
    """Plot OOD rates as barplots for multiple runs.

    Args:
        all_ood_rates: List of OOD rate data for each run
        all_labels: Labels for each run
        success_only: If True, only successful episodes were included
        smooth_window: Window size for running average smoothing (0 = no smoothing)
        num_bins: Number of bins for binned smoothing (0 = no binning)
        save_path: If provided, save figure to this path instead of displaying
        paper_mode: If True, use paper-ready styling
        max_ood_rate: If provided, set y-axis maximum to this value
        max_timesteps: If provided, only plot up to this many timesteps
    """
    filter_msg = " (success only)" if success_only else ""
    print(
        f"\nPlotting {len(all_ood_rates)} OOD rate curves for comparison{filter_msg}..."
    )

    # Set up plot style
    setup_plot_style(paper_mode=paper_mode, use_latex=True)

    # Plot all curves
    plt.figure(figsize=(10, 4.5))

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

    # Get line styles for paper mode
    line_styles = get_line_styles(len(all_ood_rates), paper_mode)

    for idx, (ood_rates, label) in enumerate(zip(all_ood_rates, all_labels)):
        # Apply max timesteps filter if specified
        if max_timesteps is not None:
            ood_rates = [d for d in ood_rates if d["timestep"] <= max_timesteps]

        timesteps = [d["timestep"] for d in ood_rates]
        rates = [d["ood_rate"] for d in ood_rates]
        stds = None  # Standard error (only computed for binning or smoothing)

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

            # Calculate mean rate and standard error for each bin
            binned_timesteps = []
            binned_rates = []
            binned_ses = []  # Standard errors
            for bin_idx in range(num_bins):
                mask = bin_indices == bin_idx
                if np.any(mask):
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    binned_timesteps.append(bin_center)
                    binned_rates.append(np.mean(rates_arr[mask]))
                    # Calculate standard error: std / sqrt(n)
                    n_samples = np.sum(mask)
                    std = np.std(rates_arr[mask])
                    se = std / np.sqrt(n_samples) if n_samples > 0 else 0
                    binned_ses.append(se)

            timesteps = binned_timesteps
            rates = binned_rates
            stds = binned_ses  # Use standard errors
        # Apply running average smoothing if requested (only if not binning)
        elif smooth_window > 1 and len(rates) >= smooth_window:
            rates_arr = np.array(rates)
            # Compute running mean
            rates_smooth = np.convolve(
                rates_arr, np.ones(smooth_window) / smooth_window, mode="valid"
            )
            # Compute running standard error
            ses_list = []
            for i in range(len(rates_smooth)):
                window_data = rates_arr[i : i + smooth_window]
                std = np.std(window_data)
                # Standard error = std / sqrt(n)
                se = std / np.sqrt(smooth_window)
                ses_list.append(se)
            stds = ses_list  # Actually standard errors
            rates = rates_smooth
            # Adjust timesteps to match the smoothed rates (center the window)
            offset = (smooth_window - 1) // 2
            timesteps = timesteps[offset : offset + len(rates)]

        # Plot the line
        plt.plot(
            timesteps,
            rates,
            label=label,
            color=colors[idx],
            linewidth=2,
            linestyle=line_styles[idx],
            alpha=0.8,
        )

        # Plot shaded region for standard error if available
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

    # Set axis limits
    if max_ood_rate is not None:
        plt.ylim(bottom=0, top=max_ood_rate)
    else:
        plt.ylim(bottom=0)

    if max_timesteps is not None:
        plt.xlim(left=0, right=max_timesteps)

    # Title only if not in paper mode
    if not paper_mode:
        title_suffix = " (Success Only)" if success_only else ""
        plt.title(f"OOD Rate Comparison by Timestep{title_suffix}")

    # Apply publication styling
    style_plot_for_publication(
        legend_outside=True,
        legend_location="center left",
        legend_bbox_to_anchor=(1.05, 0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved figure to {save_path}")
        plt.close()
    else:
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
        Tuple of (ood_rates, checkpoint_idx, level_afhp) or None if error/no data
    """
    print(f"\nLoading data from: {data_path}")

    # Load the evaluation data
    eval_data = np.load(data_path, allow_pickle=True)

    # Collect checkpoint data
    level_afhps = []
    afhps = []

    for idx, element in enumerate(eval_data["meta"]):
        level_ood_pred = element["summary"]["test"]["level_ood_pred"]
        percentage = sum(level_ood_pred) / len(level_ood_pred) * 100
        level_afhps.append(percentage)

        # Get AFHP value - prefer to use level_afhps which are more accurate
        # The afhps array sometimes contains incorrect values
        afhps.append(percentage)  # Use level AFHP as AFHP

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

        print(
            f"  Selected checkpoint {checkpoint_idx}: AFHP={actual_afhp:.2f}% (target={target_afhp:.2f}%), Performance={perf}"
        )
    else:
        # Manual selection - display all checkpoints
        print("\nCheckpoints with level AFHP percentages:")

        for idx, (level_afhp_pct, afhp) in enumerate(zip(level_afhps, afhps)):
            perf = (
                eval_data["performances"][idx]
                if idx < len(eval_data["performances"])
                else "N/A"
            )

            afhp_str = f"{afhp:.2f}" if afhp is not None else "N/A"
            print(
                f"  [{idx}] Level AFHP: {level_afhp_pct:.2f}%, AFHP: {afhp_str}, Performance: {perf}"
            )

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

    ood_rates: list[dict[str, float]] = calculate_ood_rate(
        test_summary=test_summary,
        success_only=success_only,
    )
    if len(ood_rates) == 0:
        print(f"No data available for OOD rate at checkpoint {checkpoint_idx}")
        return None
    return ood_rates, checkpoint_idx, level_afhps[checkpoint_idx]


def plot_compare_runs(
    results: dict,
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
    method_filter: Optional[List[str]] = None,
    method_exclude: Optional[List[str]] = None,
    target_afhp: Optional[float] = None,
    average_experiments: bool = False,
    save_path: Optional[str] = None,
    paper_mode: bool = False,
    max_ood_rate: Optional[float] = None,
    max_timesteps: Optional[int] = None,
):
    """Plot OOD rates for selected checkpoints from multiple runs.

    Args:
        results: Dictionary mapping method names to {exp_id: data_path}
        success_only: If True, only include episodes with reward > 0
        smooth_window: Window size for running average smoothing
        num_bins: Number of bins for binned smoothing
        method_filter: Only include these methods
        method_exclude: Exclude these methods
        target_afhp: If provided, automatically select checkpoints closest to this AFHP
        average_experiments: If True, average over all experiments instead of selecting one
        save_path: If provided, save figure to this path instead of displaying
        paper_mode: If True, use paper-ready styling
        max_ood_rate: If provided, set y-axis maximum to this value
        max_timesteps: If provided, only plot up to this many timesteps
    """
    print("\n=== Multi-Run Comparison Mode ===")

    # Apply method filter and exclusions
    methods_to_plot = sorted(results.keys())
    if method_filter:
        methods_to_plot = [
            m for m in methods_to_plot if method_is_included(m, method_filter)
        ]
        print(f"Filtering to methods: {', '.join(methods_to_plot)}")
    if method_exclude:
        methods_to_plot = [
            m for m in methods_to_plot if not method_is_filtered(m, method_exclude)
        ]
        print(f"Excluding methods: {', '.join(method_exclude)}")

    # Handle experiment selection
    if average_experiments:
        print("\n=== Averaging Over All Experiments ===")
        selected_exp = None  # We'll use all experiments
    else:
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
                    selection = input(
                        f"Select experiment (0-{len(exp_ids_list) - 1}): "
                    )
                    exp_idx = int(selection)
                    if 0 <= exp_idx < len(exp_ids_list):
                        selected_exp = exp_ids_list[exp_idx]
                        break
                    else:
                        print(
                            f"Please enter a number between 0 and {len(exp_ids_list) - 1}"
                        )
                except ValueError:
                    print("Please enter a valid number")

        print(f"\nUsing experiment {selected_exp} for all methods")

    # Get target AFHP from user if not provided
    if target_afhp is None:
        while True:
            try:
                afhp_input = input(
                    "\nEnter target AFHP percentage (e.g., 10.5 for 10.5%): "
                )
                target_afhp = float(afhp_input)
                if 0 <= target_afhp <= 100:
                    break
                else:
                    print("AFHP must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")

    print(
        f"\nSelecting checkpoints closest to AFHP={target_afhp:.2f}% for each method..."
    )

    # Collect data for each method
    all_ood_rates = []
    all_labels = []

    if average_experiments:
        # For averaging mode, collect data from all experiments per method
        for method in methods_to_plot:
            exp_data = results[method]
            method_ood_rates_by_exp = []  # Store OOD rates for each experiment

            print(f"\n--- Method: {method} ---")

            # Process all experiments for this method
            for exp_id in sorted(exp_data.keys()):
                data_path = exp_data[exp_id]

                # Load data and automatically select checkpoint based on target AFHP
                result = select_and_load_checkpoint_data(
                    run_name=f"{method}_exp{exp_id}",
                    data_path=data_path,
                    success_only=success_only,
                    target_afhp=target_afhp,
                )

                if result is not None:
                    ood_rates, checkpoint_idx, level_afhp = result
                    method_ood_rates_by_exp.append(ood_rates)

            if len(method_ood_rates_by_exp) > 0:
                # Store all experiment data for this method
                all_ood_rates.append(method_ood_rates_by_exp)
                # Use the shared label formatter (handles `_robust{200,400}`
                # suffixes and the kebab→snake METHOD_NAMES fallback).
                label = format_plot_label(
                    method, paper_mode, n_experiments=len(method_ood_rates_by_exp)
                )
                all_labels.append(label)
                print(
                    f"  Collected data from {len(method_ood_rates_by_exp)} experiments"
                )
            else:
                print("  WARNING: No valid data for any experiment")
    else:
        # Single experiment mode (original behavior)
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

            ood_rates, checkpoint_idx, level_afhp = result

            # Store data and label
            all_ood_rates.append(ood_rates)
            # Use the shared label formatter (handles `_robust{200,400}`
            # suffixes and the kebab→snake METHOD_NAMES fallback).
            base_label = format_plot_label(method, paper_mode, n_experiments=None)
            if not paper_mode:
                label = f"{base_label} (Level AFHP: {level_afhp:.1f}%)"
            else:
                label = base_label
            all_labels.append(label)

            rates = [d["ood_rate"] for d in ood_rates]
            mean_rate = np.mean(rates) if rates else 0.0
            print(f"  Mean OOD rate: {mean_rate:.2%}")

    if len(all_ood_rates) == 0:
        print("\nNo valid data collected. Exiting.")
        return

    # Plot OOD rates
    if average_experiments:
        plot_barplot_compare_aggregated(
            all_ood_rates,
            all_labels,
            success_only,
            smooth_window,
            num_bins,
            save_path,
            paper_mode,
            max_ood_rate,
            max_timesteps,
        )
    else:
        plot_barplot_compare(
            all_ood_rates,
            all_labels,
            success_only,
            smooth_window,
            num_bins,
            save_path,
            paper_mode,
            max_ood_rate,
            max_timesteps,
        )


def plot_single_run(
    results: dict,
    success_only: bool,
    smooth_window: int = 0,
    num_bins: int = 0,
    target_afhp: Optional[float] = None,
    save_path: Optional[str] = None,
    paper_mode: bool = False,
    max_ood_rate: Optional[float] = None,
    max_timesteps: Optional[int] = None,
):
    """Plot OOD rate for a single selected run and checkpoint.

    Args:
        results: Dictionary mapping method names to {exp_id: data_path}
        success_only: If True, only include episodes with reward > 0
        smooth_window: Window size for running average smoothing
        num_bins: Number of bins for binned smoothing
        target_afhp: If provided, automatically select checkpoint closest to this AFHP
        save_path: If provided, save figure to this path instead of displaying
        paper_mode: If True, use paper-ready styling
        max_ood_rate: If provided, set y-axis maximum to this value
        max_timesteps: If provided, only plot up to this many timesteps
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

    ood_rates, checkpoint_idx, level_afhp = result

    # Plot single run
    all_ood_rates = [ood_rates]
    # Format label based on paper mode
    if paper_mode:
        label = format_plot_label(selected_method, paper_mode, n_experiments=None)
    else:
        label = f"{selected_method}_exp{selected_exp} (Level AFHP: {level_afhp:.1f}%)"
    all_labels = [label]

    plot_barplot_compare(
        all_ood_rates,
        all_labels,
        success_only,
        smooth_window,
        num_bins,
        save_path,
        paper_mode,
        max_ood_rate,
        max_timesteps,
    )


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
        choices=SUPPORTED_ENVS,
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
        "--method_exclude",
        "-e",
        type=str,
        nargs="+",
        default=None,
        help="Exclude these methods in comparison mode",
    )
    parser.add_argument(
        "--target_afhp",
        type=float,
        default=None,
        help="Target AFHP percentage to automatically select closest checkpoints (e.g., 10.5 for 10.5%)",
    )
    parser.add_argument(
        "--average_experiments",
        "-a",
        action="store_true",
        help="Average over all experiments instead of selecting one",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=None,
        help="Path to save the figure (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper mode: use LaTeX rendering, small caps, and paper-ready styling",
    )
    parser.add_argument(
        "--max_ood_rate",
        type=float,
        default=None,
        help="Maximum OOD rate for y-axis (e.g., 0.1 for 10%)",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=None,
        help="Maximum number of timesteps to plot on x-axis",
    )

    args = parser.parse_args()

    # Canonicalize user-supplied method filters so kebab/snake CLI inputs both
    # match the kebab-canonical keys stored in `results`.
    if args.method_filter is not None:
        args.method_filter = [canonicalize_method(m) for m in args.method_filter]
    if args.method_exclude is not None:
        args.method_exclude = [canonicalize_method(m) for m in args.method_exclude]

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
        # Show a sample path for debugging ensemble variants
        if "ensemble" in method.lower() and exp_ids:
            sample_path = results[method][exp_ids[0]]
            print(f"    -> Sample path: {sample_path.parent.name}/{sample_path.name}")

    if args.compare_runs:
        # Multi-run comparison mode
        plot_compare_runs(
            results=results,
            success_only=args.success_only,
            smooth_window=args.smooth,
            num_bins=args.bins,
            method_filter=args.method_filter,
            method_exclude=args.method_exclude,
            target_afhp=args.target_afhp,
            average_experiments=args.average_experiments,
            save_path=args.save,
            paper_mode=args.paper,
            max_ood_rate=args.max_ood_rate,
            max_timesteps=args.max_timesteps,
        )
    else:
        # Single run mode
        plot_single_run(
            results=results,
            success_only=args.success_only,
            smooth_window=args.smooth,
            num_bins=args.bins,
            target_afhp=args.target_afhp,
            save_path=args.save,
            paper_mode=args.paper,
            max_ood_rate=args.max_ood_rate,
            max_timesteps=args.max_timesteps,
        )


if __name__ == "__main__":
    plot_ood_rate_main()

#!/usr/bin/env python3
"""
Script to plot the performance difference between coordination policy asking for help
vs running the strong policy from the start of the level.

This shows the performance penalty of switching to the strong agent mid-episode
compared to using the strong agent from the beginning.
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
import seaborn as sns
from scipy import interpolate

matplotlib.use("TkAgg")


# Method display names mapping (same as icml_plot.py)
METHOD_NAMES = {
    "max_prob": "MaxProb",
    "max_logit": "MaxLogit",
    "lb_random": "Level-Based Random",
    "ts_random": "Heuristic Strategy",
    "svdd_image": "ImageSVDD",
    "svdd_latent": "LatentSVDD",
    "ensemble": "Ensemble (multi)",
    "ensemble_single": "Ensemble",
    "latent-svdd": "Latent SVDD",
    "oc-random": "Level-Based Random",
    "wait": "Wait",
}


def parse_experiment_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse experiment directory name to extract prefix, env, and experiment ID.

    Expected format: {prefix}_{env}_exp{id}
    Examples: imcl04_coinrun_exp0, imcl04_maze_exp1

    Returns:
        Tuple of (prefix, env, exp_id) or None if pattern doesn't match
    """
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
    pattern = r"^(coinrun|maze|maze_afh|heist)_(.+)_exp(\d+)$"
    match = re.match(pattern, dir_name)
    if match:
        env = match.group(1)
        method = match.group(2)
        exp_id = int(match.group(3))
        return env, method, exp_id
    return None


def extract_strong_reval_results(
    eval_dir: Path,
    prefix_filter: Optional[List[str]] = None,
    env_filter: Optional[str] = None,
) -> Dict[str, Dict[int, Tuple[Path, Path]]]:
    """
    Extract evaluation results including strong re-evaluation files.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes (list)
        env_filter: Only include runs for this environment

    Returns:
        Dictionary mapping method names to dict of {exp_id: (original_npz, strong_reval_npz)}
    """
    results: Dict[str, Dict[int, Tuple[Path, Path]]] = defaultdict(dict)

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

            # Parse method directory name
            parsed_method = parse_method_dir(method_dir.name)
            if parsed_method is None:
                method_name = method_dir.name
            else:
                method_env, method_name, method_exp_id = parsed_method
                if method_exp_id != exp_id:
                    continue

            # Find the most recent run
            latest_run = None
            latest_time = None

            for run_dir in method_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # Check modification time
                mtime = run_dir.stat().st_mtime
                if latest_time is None or mtime > latest_time:
                    latest_time = mtime
                    latest_run = run_dir

            if latest_run is None:
                continue

            # Look for both original and strong_reval npz files
            original_npz = None
            strong_reval_npz = None

            for npz_file in latest_run.glob("*.npz"):
                if npz_file.name.endswith("_strong_reval.npz"):
                    strong_reval_npz = npz_file
                elif npz_file.name.startswith("eval_seed_") and npz_file.name.endswith("_test.npz"):
                    original_npz = npz_file

            # Only include if we have both files
            if original_npz and strong_reval_npz:
                results[method_name][exp_id] = (original_npz, strong_reval_npz)

    return dict(results)


def load_performance_data(
    original_npz: Path, strong_reval_npz: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load performance data from NPZ files.

    Returns:
        Tuple of (afhps, original_performances, performance_asked, strong_performances)
    """
    # Load original evaluation data
    orig_data = np.load(original_npz, allow_pickle=True)
    
    # Extract data
    performances = orig_data["performances"]
    meta = orig_data["meta"]
    
    # Calculate AFHPs and performance_asked for each checkpoint
    calculated_afhps = []
    performance_asked = []
    
    for idx, pt_meta in enumerate(meta):
        summary = pt_meta["summary"]["test"]
        level_seeds = summary.get("level_seeds", [])
        level_ood_pred = summary.get("level_ood_pred", [])
        raw_returns = summary.get("raw_returns", [])
        
        # Calculate AFHP as percentage of episodes where help was asked
        if len(level_ood_pred) > 0:
            afhp = sum(level_ood_pred) / len(level_ood_pred) * 100
        else:
            afhp = 0.0
        calculated_afhps.append(afhp)
        
        # Get returns only for episodes where help was asked
        asked_returns = [
            ret for seed, pred, ret in zip(level_seeds, level_ood_pred, raw_returns)
            if pred
        ]
        
        if asked_returns:
            performance_asked.append(np.mean(asked_returns))
        else:
            performance_asked.append(np.nan)
    
    # Load strong re-evaluation data
    strong_data = np.load(strong_reval_npz, allow_pickle=True)
    strong_performances = strong_data["strong_performances"]
    
    # Use calculated AFHPs instead of the ones from the file
    afhps = np.array(calculated_afhps)
    
    print(f"  AFHP range: {afhps.min():.2f}% - {afhps.max():.2f}%")
    
    return afhps, performances, np.array(performance_asked), strong_performances


def calculate_minmax_bands(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    quantiles: Tuple[float, float] = (0.25, 0.75),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate min-max quantile bands from multiple curves.
    
    Args:
        x_arrays: List of x-value arrays for each experiment
        y_arrays: List of y-value arrays for each experiment
        quantiles: Tuple of (lower_quantile, upper_quantile), default (0.25, 0.75)
        
    Returns:
        Tuple of (common_x, y_median, y_lower_quantile, y_upper_quantile)
    """
    # Collect all unique x values
    all_x_values = set()
    for x_arr in x_arrays:
        all_x_values.update(x_arr.tolist())
    common_x = np.array(sorted(all_x_values))
    
    # Create interpolation functions for each experiment
    interp_funcs = []
    for x, y in zip(x_arrays, y_arrays):
        # Remove NaN values before interpolation
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < 2:
            continue
            
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        sort_idx = np.argsort(x_valid)
        x_sorted = x_valid[sort_idx]
        y_sorted = y_valid[sort_idx]
        
        f = interpolate.interp1d(
            x_sorted, y_sorted, kind="linear", 
            bounds_error=False, fill_value=np.nan
        )
        interp_funcs.append(f)
    
    # Calculate statistics at each x value
    y_medians = []
    y_lower_quantiles = []
    y_upper_quantiles = []
    
    for x_val in common_x:
        # Collect y values from all experiments at this x
        y_values = []
        
        for f in interp_funcs:
            y_val = f(x_val)
            if not np.isnan(y_val):
                y_values.append(y_val)
        
        if len(y_values) > 0:
            # Calculate median and quantiles
            y_values = np.array(y_values)
            median = np.median(y_values)
            lower_q = np.quantile(y_values, quantiles[0])
            upper_q = np.quantile(y_values, quantiles[1])
            
            y_medians.append(median)
            y_lower_quantiles.append(lower_q)
            y_upper_quantiles.append(upper_q)
        else:
            # No values available at this x
            y_medians.append(np.nan)
            y_lower_quantiles.append(np.nan)
            y_upper_quantiles.append(np.nan)
    
    return common_x, np.array(y_medians), np.array(y_lower_quantiles), np.array(y_upper_quantiles)


def plot_strong_reval_diff(
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    env_filter: Optional[str],
    method_order: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    no_aggregate: bool = False,
    plot_absolute: bool = False,
):
    """
    Plot the performance difference between coordination policy and strong-from-start.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes
        env_filter: Only include runs for this environment
        method_order: Order of methods to plot
        method_filter: Methods to exclude
        save_path: Path to save the figure
        title: Custom title for the plot
        no_aggregate: Plot experiments separately instead of aggregating
        plot_absolute: If True, plot absolute performances instead of differences
    """
    results = extract_strong_reval_results(eval_dir, prefix_filter, env_filter)

    if not results:
        print("No results found with both original and strong re-evaluation files.")
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

    # Set up plot
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(valid_methods))

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
            original_npz, strong_reval_npz = exp_data[exp_id]
            
            try:
                afhps, performances, performance_asked, strong_performances = load_performance_data(
                    original_npz, strong_reval_npz
                )
                
                # Calculate the difference: performance_asked - strong_performances
                # This shows how much worse the coordination policy is
                if plot_absolute:
                    # Plot absolute values
                    y_coord = performance_asked
                    y_strong = strong_performances
                else:
                    # Plot difference
                    diff = performance_asked - strong_performances
                    
                    # Only keep valid (non-NaN) points
                    valid_mask = ~(np.isnan(performance_asked) | np.isnan(strong_performances))
                    if np.sum(valid_mask) > 0:
                        x_arrays.append(afhps[valid_mask])
                        y_arrays.append(diff[valid_mask])
                
            except Exception as e:
                print(f"Warning: Failed to load data for {method} exp{exp_id}: {e}")
                continue

        if len(x_arrays) == 0:
            print(f"Warning: No valid data for {method}, skipping...")
            continue

        # Get display name
        label = METHOD_NAMES.get(method, method)

        if len(x_arrays) == 1 or no_aggregate:
            # Single experiment or no aggregation mode
            if no_aggregate and len(x_arrays) > 1:
                # Plot each experiment separately
                base_color = colors[method_idx]
                for i, (x, y, exp_id) in enumerate(zip(x_arrays, y_arrays, exp_ids)):
                    sort_idx = np.argsort(x)
                    alpha = 0.7 + (i / len(x_arrays)) * 0.3
                    exp_label = f"{label} exp{exp_id}"
                    
                    plt.plot(
                        x[sort_idx],
                        y[sort_idx],
                        label=exp_label,
                        color=base_color,
                        alpha=alpha,
                        marker="o" if method == "wait" else None,
                        markersize=3 if method == "wait" else None,
                        linewidth=1.5,
                    )
            else:
                # Single experiment
                x, y = x_arrays[0], y_arrays[0]
                sort_idx = np.argsort(x)
                plt.plot(
                    x[sort_idx],
                    y[sort_idx],
                    label=f"{label} (n=1)",
                    color=colors[method_idx],
                    marker="o" if method == "wait" else None,
                    markersize=4,
                )
        else:
            # Multiple experiments, aggregate using quantile bands
            common_x, y_median, y_lower_q, y_upper_q = calculate_minmax_bands(
                x_arrays, y_arrays, quantiles=(0.25, 0.75)
            )
            
            # Filter out NaN values
            valid_mask = ~np.isnan(y_median)
            common_x = common_x[valid_mask]
            y_median = y_median[valid_mask]
            y_lower_q = y_lower_q[valid_mask]
            y_upper_q = y_upper_q[valid_mask]
            
            n_exps = len(x_arrays)
            
            # Plot median line
            plt.plot(
                common_x,
                y_median,
                label=f"{label} (n={n_exps})",
                color=colors[method_idx],
                linewidth=2,
            )
            
            # Plot quantile band
            plt.fill_between(
                common_x,
                y_lower_q,
                y_upper_q,
                color=colors[method_idx],
                alpha=0.2,
            )

    # Add reference line at y=0
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="No difference")

    # Labels and title
    env_str = env_filter if env_filter else "all"
    prefix_str = ",".join(prefix_filter) if prefix_filter else "all"

    plt.xlabel("Ask-For-Help Percentage (AFHP)")
    
    if plot_absolute:
        plt.ylabel("Average Return")
        default_title = f"Coordination vs Strong-from-Start Performance ({env_str}, prefix={prefix_str})"
    else:
        plt.ylabel("Performance Difference (Coordination - Strong from Start)")
        default_title = f"Performance Loss from Mid-Episode Switching ({env_str}, prefix={prefix_str})"

    if title:
        plt.title(title)
    else:
        plt.title(default_title)
    
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance difference between coordination policy and strong-from-start"
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
        "--save",
        type=str,
        default=None,
        help="Path to save the figure (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Plot experiments separately instead of aggregating",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Plot absolute performances instead of differences",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Parse method order
    method_order = None
    if args.method_order:
        method_order = [m.strip() for m in args.method_order.split(",")]

    plot_strong_reval_diff(
        eval_dir=eval_dir,
        prefix_filter=args.prefix,
        env_filter=args.env,
        method_order=method_order,
        method_filter=args.method_filter,
        save_path=args.save,
        title=args.title,
        no_aggregate=args.no_aggregate,
        plot_absolute=args.absolute,
    )


if __name__ == "__main__":
    main()
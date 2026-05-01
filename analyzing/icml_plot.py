#!/usr/bin/env python3
"""
Script to plot ICML evaluation results with aggregation across experiments.

Handles directory structure like:
    imcl04_coinrun_exp0/
    imcl04_coinrun_exp1/
    imcl04_maze_exp0/
    imcl04_robust400_maze_exp0/
    ...

Aggregates results for the same method across different experiment IDs,
plotting mean with shaded standard error regions.
Robust eval groups are plotted as distinct method variants.
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
from scipy import interpolate, integrate

from analyzing.utils import extract_x_and_y_values
from analyzing.plotting_common import (
    METHOD_NAMES,
    setup_plot_style,
    get_line_styles,
    style_plot_for_publication,
)

matplotlib.use("TkAgg")

SUPPORTED_ENVS = [
    "coinrun",
    "coinrun_proxy_fail",
    "maze",
    "maze_afh",
    "maze_proxy_fail",
    "heist",
]
ENV_PATTERN = "|".join(
    re.escape(env) for env in sorted(SUPPORTED_ENVS, key=len, reverse=True)
)
ROBUST_VARIANTS = ("robust200", "robust400")
ROBUST_LABELS = {
    "robust200": "Robust 200M",
    "robust400": "Robust 400M",
}
ROBUST_PATTERN = "|".join(re.escape(variant) for variant in ROBUST_VARIANTS)

# Data key display names mapping
DATA_KEY_NAMES = {
    "step_afhp": "Ask-For-Help Percentage (AFHP, per timestep)",
    "level_afhp": "Ask-For-Help Percentage (AFHP)",
    "performance": "Average Return",
    "performance_asked": "Average Reward (Asked for Help)",
    "performance_not_asked": "Average Reward (Did Not Ask)",
    "performance_asked_correctly": "Average Reward (True Positive)",
    "performance_not_asked_correctly": "Average Reward (True Negative)",
    "performance_not_asked_incorrectly": "Average Reward (False Negative)",
    "ood_accuracy": "OOD Accuracy",
    "true_positive": "True Positive Rate",
    "false_positive": "False Positive Rate",
    "true_negative": "True Negative Rate",
    "false_negative": "False Negative Rate",
    "episode_length_mean": "Mean Episode Length",
    "episode_length_success_mean": "Mean Episode Length (Success)",
    "first_ood_timestep_mean": "Mean First OOD Timestep",
}

# Keys that filter by asking behavior (need special handling for reference lines)
FILTERED_PERFORMANCE_KEYS = {
    "performance_asked",
    "performance_not_asked",
    "performance_asked_correctly",
    "performance_not_asked_correctly",
    "performance_not_asked_incorrectly",
}


def parse_robust_experiment_dir(
    dir_name: str,
) -> Optional[Tuple[str, str, str, int]]:
    """
    Parse robust experiment directory names.

    Expected format: {prefix}_robust{200,400}_{env}_exp{id}
    Example: icml04_robust400_maze_exp0

    Returns:
        Tuple of (base_prefix, robust_variant, env, exp_id), or None.
    """
    pattern = rf"^(.+)_({ROBUST_PATTERN})_({ENV_PATTERN})_exp(\d+)$"
    match = re.match(pattern, dir_name)
    if match:
        prefix = match.group(1)
        robust_variant = match.group(2)
        env = match.group(3)
        exp_id = int(match.group(4))
        return prefix, robust_variant, env, exp_id
    return None


def parse_experiment_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse experiment directory name to extract prefix, env, and experiment ID.

    Expected format: {prefix}_{env}_exp{id}
    Examples: imcl04_coinrun_exp0, imcl04_maze_proxy_fail_exp1

    Returns:
        Tuple of (prefix, env, exp_id) or None if pattern doesn't match
    """
    robust_parsed = parse_robust_experiment_dir(dir_name)
    if robust_parsed is not None:
        prefix, _, env, exp_id = robust_parsed
        return prefix, env, exp_id

    pattern = rf"^(.+)_({ENV_PATTERN})_exp(\d+)$"
    match = re.match(pattern, dir_name)
    if match:
        prefix = match.group(1)
        env = match.group(2)
        exp_id = int(match.group(3))
        return prefix, env, exp_id
    return None


def split_robust_method(method: str) -> Tuple[str, Optional[str]]:
    """Split method names like max-prob_robust400 into base method and variant."""
    for robust_variant in ROBUST_VARIANTS:
        suffix = f"_{robust_variant}"
        if method.endswith(suffix):
            return method[: -len(suffix)], robust_variant
    return method, None


def add_robust_suffix(method: str, robust_variant: Optional[str]) -> str:
    """Attach robust variant to method name unless it is already present."""
    if robust_variant is None:
        return method
    _, existing_variant = split_robust_method(method)
    if existing_variant is not None:
        return method
    return f"{method}_{robust_variant}"


def method_display_name(method: str) -> str:
    """Return a display name, accepting both hyphen and underscore method ids."""
    return METHOD_NAMES.get(method, METHOD_NAMES.get(method.replace("-", "_"), method))


def format_plot_label(
    method: str, paper_mode: bool, n_experiments: Optional[int] = None
) -> str:
    """Format method labels, including robust strong-policy variants."""
    base_method, robust_variant = split_robust_method(method)
    label = method_display_name(base_method)

    if robust_variant is not None:
        label = f"{label} ({ROBUST_LABELS[robust_variant]})"

    if not paper_mode and n_experiments is not None:
        label = f"{label} (n={n_experiments})"

    return label


def prefix_matches_filter(
    prefix: str,
    robust_variant: Optional[str],
    prefix_filter: Optional[List[str]],
) -> bool:
    """Match both base prefixes and robust output prefixes."""
    if prefix_filter is None:
        return True

    candidates = [prefix]
    if robust_variant is not None:
        candidates.append(f"{prefix}_{robust_variant}")

    return any(candidate in prefix_filter for candidate in candidates)


def expand_method_order_for_robust_variants(
    method_order: List[str], available_methods: Dict[str, Dict[int, Path]]
) -> List[str]:
    """Place robust variants after their base method when a base order is provided."""
    expanded_order = []
    seen = set()

    for method in method_order:
        candidate_methods = [method]
        candidate_methods.extend(
            f"{method}_{robust_variant}" for robust_variant in ROBUST_VARIANTS
        )

        for candidate_method in candidate_methods:
            if candidate_method in available_methods and candidate_method not in seen:
                expanded_order.append(candidate_method)
                seen.add(candidate_method)

    return expanded_order


def method_is_filtered(method: str, method_filter: List[str]) -> bool:
    """Filter robust method variants when either full or base method is filtered."""
    base_method, _ = split_robust_method(method)
    return method in method_filter or base_method in method_filter


def parse_method_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse method directory name to extract env, method, and experiment ID.

    Expected format: {env}_{method}_exp{id}
    Examples: coinrun_max_prob_exp0, maze_proxy_fail_max_prob_exp1

    Returns:
        Tuple of (env, method, exp_id) or None if pattern doesn't match
    """
    pattern = rf"^({ENV_PATTERN})_(.+)_exp(\d+)$"
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

        robust_parsed = parse_robust_experiment_dir(child.name)
        robust_variant = None
        if robust_parsed is not None:
            prefix, robust_variant, env, exp_id = robust_parsed
        else:
            parsed = parse_experiment_dir(child.name)
            if parsed is None:
                continue
            prefix, env, exp_id = parsed

        if not prefix_matches_filter(prefix, robust_variant, prefix_filter):
            continue

        # Apply filters
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
            method_name = add_robust_suffix(method_name, robust_variant)

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


def calculate_minmax_bands(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    quantiles: Tuple[float, float] = (0.25, 0.75),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate min-max quantile bands from multiple curves.
    
    For each x value, interpolate all curves to that x, then calculate
    the median and specified quantiles across experiments.
    
    Args:
        x_arrays: List of x-value arrays for each experiment
        y_arrays: List of y-value arrays for each experiment
        quantiles: Tuple of (lower_quantile, upper_quantile), default (0.25, 0.75) for IQR
        
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
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
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


def calculate_auc_with_bands(
    x: np.ndarray, 
    y_median: np.ndarray, 
    y_lower: np.ndarray, 
    y_upper: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate area under the curve (AUC) for median and quantile bands.
    
    Args:
        x: Common x values
        y_median: Median y values
        y_lower: Lower quantile y values  
        y_upper: Upper quantile y values
        
    Returns:
        Tuple of (auc_median, auc_lower, auc_upper)
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(y_median) | np.isnan(y_lower) | np.isnan(y_upper))
    if np.sum(valid_mask) < 2:
        return np.nan, np.nan, np.nan
    
    x_valid = x[valid_mask]
    y_median_valid = y_median[valid_mask]
    y_lower_valid = y_lower[valid_mask]
    y_upper_valid = y_upper[valid_mask]
    
    # Calculate AUC using trapezoidal rule
    auc_median = integrate.trapezoid(y_median_valid, x_valid)
    auc_lower = integrate.trapezoid(y_lower_valid, x_valid)
    auc_upper = integrate.trapezoid(y_upper_valid, x_valid)
    
    return auc_median, auc_lower, auc_upper


def print_auc_latex_table(method_auc_data: Dict[str, Tuple[float, float, float]], 
                         x_label: str, y_label: str,
                         normalize_by_range: bool = True,
                         x_range: Optional[Tuple[float, float]] = None,
                         weak_performance: Optional[float] = None,
                         strong_performance: Optional[float] = None):
    """
    Print AUC results as a LaTeX-compatible table.
    
    Args:
        method_auc_data: Dictionary mapping method names to (auc_median, auc_lower, auc_upper)
        x_label: Label for x-axis (for table caption)
        y_label: Label for y-axis (for table caption)
        normalize_by_range: If True, normalize AUC by x-axis range
        x_range: Optional x-axis range for normalization
        weak_performance: Weak (novice) agent performance for y-axis normalization
        strong_performance: Strong (expert) agent performance for y-axis normalization
    """
    print("\n" + "="*80)
    print("AREA UNDER THE CURVE (AUC) RESULTS")
    print("="*80)
    
    # Sort methods by median AUC (descending)
    sorted_methods = sorted(method_auc_data.items(), 
                          key=lambda x: x[1][0] if not np.isnan(x[1][0]) else -np.inf, 
                          reverse=True)
    
    # Print LaTeX table
    print("\n% LaTeX Table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("Method & AUC (Median [IQR]) \\\\")
    print("\\midrule")
    
    for method, (auc_median, auc_lower, auc_upper) in sorted_methods:
        display_name = format_plot_label(method, paper_mode=True)
        
        if np.isnan(auc_median):
            print(f"{display_name} & -- \\\\")
        else:
            # First normalize by performance range (weak to strong)
            if weak_performance is not None and strong_performance is not None:
                perf_range = strong_performance - weak_performance
                if abs(perf_range) > 1e-6:  # Avoid division by zero
                    # Subtract baseline (weak * x_range) and normalize
                    if x_range is not None:
                        baseline_area = weak_performance * (x_range[1] - x_range[0])
                        auc_median = (auc_median - baseline_area) / perf_range
                        auc_lower = (auc_lower - baseline_area) / perf_range
                        auc_upper = (auc_upper - baseline_area) / perf_range
            
            # Then normalize by x-axis range if requested
            if normalize_by_range and x_range is not None:
                range_width = x_range[1] - x_range[0]
                if range_width > 0:
                    auc_median /= range_width
                    auc_lower /= range_width
                    auc_upper /= range_width
            
            # Format as median [lower, upper]
            print(f"{display_name} & {auc_median:.3f} [{auc_lower:.3f}, {auc_upper:.3f}] \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Area Under the Curve (AUC) for {y_label} vs {x_label}.")
    if weak_performance is not None and strong_performance is not None:
        print(" Values are normalized such that 0 = novice performance and 1 = expert performance.")
    if normalize_by_range:
        print(" Values are also normalized by the x-axis range.")
    print(" IQR shows 25th-75th percentile range.}")
    print("\\label{tab:auc_results}")
    print("\\end{table}")
    
    # Also print human-readable version
    print("\n" + "-"*60)
    print("Human-readable version:")
    print("-"*60)
    print(f"{'Method':<30} {'AUC Median':<15} {'AUC IQR':<25}")
    print("-"*60)
    
    for method, (auc_median, auc_lower, auc_upper) in sorted_methods:
        display_name = (
            format_plot_label(method, paper_mode=True)
            .replace(r"\textsc{", "")
            .replace("}", "")
        )
        
        if np.isnan(auc_median):
            print(f"{display_name:<30} {'N/A':<15}")
        else:
            # First normalize by performance range (weak to strong)
            if weak_performance is not None and strong_performance is not None:
                perf_range = strong_performance - weak_performance
                if abs(perf_range) > 1e-6:  # Avoid division by zero
                    # Subtract baseline (weak * x_range) and normalize
                    if x_range is not None:
                        baseline_area = weak_performance * (x_range[1] - x_range[0])
                        auc_median = (auc_median - baseline_area) / perf_range
                        auc_lower = (auc_lower - baseline_area) / perf_range
                        auc_upper = (auc_upper - baseline_area) / perf_range
            
            # Then normalize by x-axis range if requested
            if normalize_by_range and x_range is not None:
                range_width = x_range[1] - x_range[0]
                if range_width > 0:
                    auc_median /= range_width
                    auc_lower /= range_width
                    auc_upper /= range_width
                    
            print(f"{display_name:<30} {auc_median:<15.3f} [{auc_lower:.3f}, {auc_upper:.3f}]")
    
    print("-"*60)


def plot_icml_results(
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    env_filter: Optional[str],
    x_data_key: str,
    y_data_key: str,
    method_order: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    use_stderr: bool = True,
    disable_horizontal_lines: bool = False,
    disable_random_line: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    no_aggregate: bool = False,
    paper_mode: bool = False,
    calculate_auc: bool = False,
):
    """
    Plot ICML results with aggregation across experiments.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes (list)
        env_filter: Only include runs for this environment
        x_data_key: Key for x-axis data
        y_data_key: Key for y-axis data
        method_order: Order of methods to plot
        method_filter: Methods to exclude
        use_stderr: If True, use standard error; otherwise use standard deviation
        disable_horizontal_lines: Disable weak/oracle reference lines
        disable_random_line: Disable random baseline diagonal line
        save_path: Path to save the figure
        title: Custom title for the plot (overrides auto-generated title)
        no_aggregate: Plot experiments separately instead of aggregating
        paper_mode: If True, remove title and n=X from labels for paper figures
        calculate_auc: If True, calculate and display AUC for each method
    """
    results = extract_icml_results(eval_dir, prefix_filter, env_filter)

    if not results:
        print("No results found matching the filters.")
        return

    # Determine method order
    if method_order is None:
        method_order = sorted(results.keys())
    else:
        method_order = expand_method_order_for_robust_variants(method_order, results)

    if method_filter is not None:
        method_order = [
            m for m in method_order if not method_is_filtered(m, method_filter)
        ]

    # Filter to valid methods
    valid_methods = [m for m in method_order if m in results]

    if not valid_methods:
        print("No valid methods found to plot.")
        return

    # Set up plot style
    setup_plot_style(paper_mode=paper_mode, use_latex=True)
    
    plt.figure(figsize=(8, 4.5))
    colors = sns.color_palette("husl", len(valid_methods))
    
    # Get line styles for paper mode
    line_styles = get_line_styles(len(valid_methods), paper_mode, valid_methods)

    # Store weak/oracle performance for reference lines
    all_first_performances = []
    all_last_performances = []
    # Track unfiltered weak agent performance (for performance_asked)
    all_weak_performances = []
    # Track x range for random baseline line
    all_x_min = []
    all_x_max = []
    
    # Store AUC data for each method
    method_auc_data = {}
    overall_x_min = float('inf')
    overall_x_max = float('-inf')

    for method_idx, method in enumerate(valid_methods):
        exp_data = results[method]
        exp_ids = sorted(exp_data.keys())

        if len(exp_ids) == 0:
            print(f"Warning: No experiments found for {method}, skipping...")
            continue

        # Load all experiment data
        x_arrays = []
        y_arrays = []
        meta_arrays = []  # Store meta data for confidence interval calculation

        for exp_id in exp_ids:
            data_path = exp_data[exp_id]
            try:
                eval_data = np.load(data_path, allow_pickle=True)
                x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

                if len(x) > 0:
                    x_arrays.append(x)
                    y_arrays.append(y)
                    
                    # Store meta data if available
                    if 'meta' in eval_data:
                        meta_arrays.append(eval_data['meta'])
                    else:
                        meta_arrays.append([])

                    # Track first/last for reference lines
                    all_first_performances.append(y[0])
                    all_last_performances.append(y[-1])
                    # Track x range
                    all_x_min.append(x.min())
                    all_x_max.append(x.max())

                    # For filtered performance keys, also get unfiltered weak performance
                    if y_data_key in FILTERED_PERFORMANCE_KEYS:
                        _, y_unfiltered = extract_x_and_y_values(
                            eval_data, x_data_key, "performance"
                        )
                        if len(y_unfiltered) > 0:
                            all_weak_performances.append(y_unfiltered[0])
            except Exception as e:
                print(f"Warning: Failed to load {data_path}: {e}")
                continue

        if len(x_arrays) == 0:
            print(f"Warning: No valid data for {method}, skipping...")
            continue
        
        # Print AFHP values for wait policy
        if method == "wait" and x_data_key in ["step_afhp", "level_afhp"]:
            print(f"\n=== Wait Policy AFHP Values ===")
            for exp_idx, x_values in enumerate(x_arrays):
                print(f"Experiment {exp_ids[exp_idx]}: {sorted(set(x_values))}")
            all_afhp_values = set()
            for x in x_arrays:
                all_afhp_values.update(x)
            print(f"All unique AFHP values across experiments: {sorted(all_afhp_values)}")
            print(f"Total unique values: {len(all_afhp_values)}")

        if len(x_arrays) == 1 or no_aggregate:
            # Single experiment or no aggregation mode
            if no_aggregate and len(x_arrays) > 1:
                # Plot each experiment separately with slightly different shades
                base_color = colors[method_idx]
                for i, (x, y, exp_id) in enumerate(zip(x_arrays, y_arrays, exp_ids)):
                    sort_idx = np.argsort(x)
                    # Vary alpha or lightness for different experiments
                    alpha = 0.7 + (i / len(x_arrays)) * 0.3
                    base_label = format_plot_label(
                        method, paper_mode, n_experiments=None
                    )
                    exp_label = f"{base_label} exp{exp_id}"
                    
                    # Add markers for wait policy
                    if method == "wait":
                        plt.plot(
                            x[sort_idx],
                            y[sort_idx],
                            label=exp_label,
                            color=base_color,
                            alpha=alpha,
                            marker="o",
                            markersize=3,
                            linewidth=1.5,
                            linestyle=line_styles[method_idx],
                        )
                    else:
                        plt.plot(
                            x[sort_idx],
                            y[sort_idx],
                            label=exp_label,
                            color=base_color,
                            alpha=alpha,
                            linewidth=1.5,
                            linestyle=line_styles[method_idx],
                        )
            else:
                # Single experiment
                x, y = x_arrays[0], y_arrays[0]
                sort_idx = np.argsort(x)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                
                # Format label using shared function
                plot_label = format_plot_label(
                    method,
                    paper_mode,
                    n_experiments=1 if not paper_mode else None,
                )
                plt.plot(
                    x_sorted,
                    y_sorted,
                    label=plot_label,
                    color=colors[method_idx],
                    marker="o" if method == "wait" else None,
                    markersize=4,
                    linestyle=line_styles[method_idx],
                )
                
                # Calculate AUC for single experiment
                if calculate_auc:
                    # For single experiment, use the same value for all bands
                    auc = integrate.trapezoid(y_sorted, x_sorted)
                    method_auc_data[method] = (auc, auc, auc)
                    
                    # Update overall x-range
                    if len(x_sorted) > 0:
                        overall_x_min = min(overall_x_min, x_sorted.min())
                        overall_x_max = max(overall_x_max, x_sorted.max())
        else:
            # Multiple experiments, aggregate using quantile bands
            # Use 25th-75th percentile bands (interquartile range)
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
            
            # Format label using shared function
            plot_label = format_plot_label(
                method, paper_mode, n_experiments=n_exps
            )
            
            # Plot median line
            plt.plot(
                common_x,
                y_median,
                label=plot_label,
                color=colors[method_idx],
                linewidth=2,
                linestyle=line_styles[method_idx],
            )
            
            # Plot quantile band (25th-75th percentile)
            plt.fill_between(
                common_x,
                y_lower_q,
                y_upper_q,
                color=colors[method_idx],
                alpha=0.2,
            )
            
            # Calculate AUC if requested
            if calculate_auc:
                auc_median, auc_lower, auc_upper = calculate_auc_with_bands(
                    common_x, y_median, y_lower_q, y_upper_q
                )
                method_auc_data[method] = (auc_median, auc_lower, auc_upper)
                
                # Update overall x-range
                if len(common_x) > 0:
                    overall_x_min = min(overall_x_min, common_x.min())
                    overall_x_max = max(overall_x_max, common_x.max())

    # Add reference lines
    if not disable_horizontal_lines and all_first_performances:
        mean_first = np.mean(all_first_performances)
        mean_last = np.mean(all_last_performances)

        # For filtered performance keys, use unfiltered weak performance
        if y_data_key in FILTERED_PERFORMANCE_KEYS and all_weak_performances:
            weak_perf = np.mean(all_weak_performances)
        else:
            weak_perf = mean_first

        plt.axhline(
            y=weak_perf,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=r"\textsc{Novice}",
        )
        plt.axhline(
            y=mean_last,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=r"\textsc{Expert}",
        )

    # Add random baseline diagonal line (from weak to oracle)
    if not disable_random_line and all_first_performances and all_x_min:
        mean_last = np.mean(all_last_performances)
        x_min = min(all_x_min)
        x_max = max(all_x_max)

        # For filtered performance keys, use unfiltered weak performance
        if y_data_key in FILTERED_PERFORMANCE_KEYS and all_weak_performances:
            random_start = np.mean(all_weak_performances)
        else:
            random_start = np.mean(all_first_performances)

        # Use a unique line style for random that's different from all methods
        # Since methods get assigned styles starting from index 0, we'll use a later style
        random_linestyle = (0, (2, 2, 10, 2))  # Custom pattern: short dash, short dash, long gap, short dash
        if paper_mode:
            # In paper mode, ensure random gets a distinct style
            random_linestyle = (0, (2, 2, 10, 2))
        else:
            random_linestyle = ":"  # Keep dotted in non-paper mode
            
        plt.plot(
            [x_min, x_max],
            [random_start, mean_last],
            color="black",
            linestyle=random_linestyle,
            alpha=0.9,
            linewidth=2,
            label=r"\textsc{Random}",
            zorder=10,  # Draw on top
        )

    # Labels and title
    env_str = env_filter if env_filter else "all"
    prefix_str = ",".join(prefix_filter) if prefix_filter else "all"
    # Update error type to reflect quantile bands
    error_type = "IQR"  # Interquartile range

    # Get display names for axis labels
    x_label = DATA_KEY_NAMES.get(x_data_key, x_data_key)
    y_label = DATA_KEY_NAMES.get(y_data_key, y_data_key)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Use custom title if provided, otherwise generate one (skip if paper_mode)
    if not paper_mode:
        if title:
            plt.title(title)
        else:
            plt.title(
                f"{y_label} vs {x_label} ({env_str}, prefix={prefix_str}, shaded={error_type})"
            )
    # Apply publication styling
    style_plot_for_publication(
        legend_outside=True,
        legend_location='center left',
        legend_bbox_to_anchor=(1.05, 0.5)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    # Print AUC table if requested
    if calculate_auc and method_auc_data:
        x_range = (overall_x_min, overall_x_max) if overall_x_min != float('inf') else None
        
        # Calculate weak and strong performance for normalization
        weak_perf = None
        strong_perf = None
        
        if all_first_performances:
            # For filtered performance keys, use unfiltered weak performance
            if y_data_key in FILTERED_PERFORMANCE_KEYS and all_weak_performances:
                weak_perf = np.mean(all_weak_performances)
            else:
                weak_perf = np.mean(all_first_performances)
        
        if all_last_performances:
            strong_perf = np.mean(all_last_performances)
        
        print_auc_latex_table(method_auc_data, x_label, y_label, 
                            normalize_by_range=True, x_range=x_range,
                            weak_performance=weak_perf,
                            strong_performance=strong_perf)


def list_available_methods(
    eval_dir: Path, prefix_filter: Optional[List[str]], env_filter: Optional[str]
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
        "--x_data_key",
        type=str,
        default="step_afhp",
        help="Key for the x data (default: step_afhp)",
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
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot (overrides auto-generated title)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available methods and exit",
    )
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Plot experiments separately instead of aggregating",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper mode: remove title and n=X from labels for cleaner figures",
    )
    parser.add_argument(
        "--auc",
        action="store_true",
        help="Calculate and display area under the curve (AUC) for each method with IQR bands",
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
        title=args.title,
        no_aggregate=args.no_aggregate,
        paper_mode=args.paper,
        calculate_auc=args.auc,
    )


if __name__ == "__main__":
    main()

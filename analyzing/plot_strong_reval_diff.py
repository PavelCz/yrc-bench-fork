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
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import numpy as np
from scipy import interpolate

from analyzing.plotting_common import (
    format_label,
    get_line_styles,
    setup_plot_style,
    style_plot_for_publication,
)

plt = None
sns = None


def configure_matplotlib_backend(save_path: Optional[str]) -> None:
    """Select a backend before pyplot or seaborn is imported."""
    if save_path:
        matplotlib.use("Agg", force=True)
        return

    try:
        matplotlib.use("TkAgg", force=True)
    except ImportError as exc:
        raise RuntimeError(
            "Could not load Matplotlib's TkAgg backend for interactive display. "
            "Use --save PATH, or verify that Tk works with: "
            'python -c \'import matplotlib; matplotlib.use("TkAgg", force=True); '
            "import matplotlib.pyplot as plt; print(matplotlib.get_backend())'"
        ) from exc


# Method display names mapping (same as paper_plot.py)
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
    Examples: dummy04_coinrun_exp0, dummy04_maze_exp1

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
    exp_id_filter: Optional[Set[int]] = None,
) -> Dict[str, Dict[int, Tuple[Path, Path]]]:
    """
    Extract evaluation results including strong re-evaluation files.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes (list)
        env_filter: Only include runs for this environment
        exp_id_filter: Only include these experiment IDs

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
        if exp_id_filter is not None and exp_id not in exp_id_filter:
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

            # Find the most recent run that contains both required artifacts.
            # The newest directory can be a partial sync or an original eval-only run.
            original_npz = None
            strong_reval_npz = None
            for run_dir in method_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                run_original_npz = None
                run_strong_reval_npz = None
                for npz_file in run_dir.glob("*.npz"):
                    if npz_file.name.endswith("_strong_reval.npz"):
                        run_strong_reval_npz = npz_file
                    elif npz_file.name.startswith(
                        "eval_seed_"
                    ) and npz_file.name.endswith("_test.npz"):
                        run_original_npz = npz_file

                if run_original_npz is None or run_strong_reval_npz is None:
                    continue

                if (
                    original_npz is None
                    or run_dir.stat().st_mtime > original_npz.parent.stat().st_mtime
                ):
                    original_npz = run_original_npz
                    strong_reval_npz = run_strong_reval_npz

            # Only include if we have both files
            if original_npz and strong_reval_npz:
                results[method_name][exp_id] = (original_npz, strong_reval_npz)

    return dict(results)


def load_performance_data(
    original_npz: Path, strong_reval_npz: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load performance data from NPZ files.

    Returns:
        Tuple of (afhps, original_performances, performance_asked,
                  strong_performances, weak_on_help_performances).

        `weak_on_help_performances[i]` is the mean weak-agent return on the
        episodes where CurvePoint `i` asked for help. The weak per-episode
        returns come from the AFHP=0 (no-help) CurvePoint inside
        `original_npz`; the mask is `meta[i].summary.test.level_ood_pred`.
        Entries are NaN when no AFHP=0 baseline is recoverable, when the
        per-CurvePoint help mask is empty, or when array shapes disagree.
    """
    orig_data = np.load(original_npz, allow_pickle=True)

    performances = orig_data["performances"]
    meta = orig_data["meta"]

    # Load strong re-evaluation data early so we can use its aligned
    # original_help_performances field when present.
    strong_data = np.load(strong_reval_npz, allow_pickle=True)
    strong_performances = np.asarray(strong_data["strong_performances"], dtype=float)
    saved_original_help = strong_data.get("original_help_performances", None)

    calculated_afhps = []
    performance_asked = []

    for idx, pt_meta in enumerate(meta):
        summary = pt_meta["summary"]["test"]
        level_ood_pred = summary.get("level_ood_pred", [])
        raw_returns = summary.get("raw_returns", [])

        # Calculate AFHP as percentage of episodes where help was asked
        if len(level_ood_pred) > 0:
            afhp = sum(level_ood_pred) / len(level_ood_pred) * 100
        else:
            afhp = 0.0
        calculated_afhps.append(afhp)

        if saved_original_help is not None and idx < len(saved_original_help):
            performance_asked.append(float(saved_original_help[idx]))
        else:
            # Get returns only for episodes where help was asked
            asked_returns = [
                ret for pred, ret in zip(level_ood_pred, raw_returns) if pred
            ]

            if asked_returns:
                performance_asked.append(np.mean(asked_returns))
            else:
                performance_asked.append(np.nan)

    # CRITICAL VALIDATION: Ensure arrays are properly aligned
    print("\n  === Array Length Validation ===")
    print("  Original data:")
    print(f"    - Number of checkpoints (meta): {len(meta)}")
    print(f"    - Number of performances: {len(performances)}")
    print(f"    - Number of calculated AFHPs: {len(calculated_afhps)}")
    print("  Strong re-eval data:")
    print(f"    - Number of strong performances: {len(strong_performances)}")

    # Check if arrays have the same length
    if len(performances) != len(strong_performances):
        print("  ERROR: Performance array length mismatch!")
        print(f"    Original: {len(performances)}, Strong: {len(strong_performances)}")
        raise ValueError("Cannot compare performances - arrays have different lengths!")

    if len(calculated_afhps) != len(strong_performances):
        print("  ERROR: AFHP/performance array length mismatch!")
        print(
            f"    AFHPs: {len(calculated_afhps)}, Strong performances: {len(strong_performances)}"
        )
        raise ValueError("Cannot align AFHPs with performances!")

    # Calculate AFHPs from strong re-evaluation data for validation
    strong_meta = strong_data.get("meta", [])
    strong_calculated_afhps = []

    if len(strong_meta) > 0:
        for idx, pt_meta in enumerate(strong_meta):
            summary = pt_meta["summary"]["test"]
            level_ood_pred = summary.get("level_ood_pred", [])

            # Calculate AFHP from strong data
            if len(level_ood_pred) > 0:
                afhp = sum(level_ood_pred) / len(level_ood_pred) * 100
            else:
                afhp = 0.0
            strong_calculated_afhps.append(afhp)

        # Validate that AFHPs match between original and strong data
        if len(calculated_afhps) == len(strong_calculated_afhps):
            afhp_diff = np.array(calculated_afhps) - np.array(strong_calculated_afhps)
            max_diff = np.abs(afhp_diff).max()

            if max_diff > 0.01:  # Allow small floating point differences
                print("  WARNING: AFHP mismatch between original and strong data!")
                print(f"  Maximum difference: {max_diff:.4f}%")
                print(f"  Original AFHPs: {calculated_afhps[:5]}... (showing first 5)")
                print(
                    f"  Strong AFHPs:   {strong_calculated_afhps[:5]}... (showing first 5)"
                )
            else:
                print(
                    f"  ✓ AFHPs match between original and strong data (max diff: {max_diff:.6f}%)"
                )
        else:
            print(
                f"  WARNING: Different number of checkpoints in original ({len(calculated_afhps)}) vs strong ({len(strong_calculated_afhps)}) data!"
            )
    else:
        print("  INFO: No meta data in strong re-eval file to validate AFHPs")

    # Also check AFHPs from stored arrays if they exist
    orig_stored_afhps = orig_data.get("afhps", None)
    if orig_stored_afhps is not None and len(orig_stored_afhps) > 0:
        print(
            f"  Original stored AFHP range: {orig_stored_afhps.min():.2f}% - {orig_stored_afhps.max():.2f}%"
        )
        # Compare stored vs calculated
        if len(orig_stored_afhps) == len(calculated_afhps):
            stored_vs_calc_diff = np.abs(
                orig_stored_afhps - np.array(calculated_afhps)
            ).max()
            if stored_vs_calc_diff > 0.01:
                print(
                    f"  WARNING: Stored AFHPs differ from calculated AFHPs in original data (max diff: {stored_vs_calc_diff:.2f}%)"
                )
                # Show some examples
                n_examples = min(5, len(orig_stored_afhps))
                print(f"  Examples (first {n_examples}):")
                print(f"    Stored:     {orig_stored_afhps[:n_examples]}")
                print(f"    Calculated: {calculated_afhps[:n_examples]}")
                # Check if it's a factor of 100 issue
                factor_check = np.array(calculated_afhps[:n_examples]) / (
                    orig_stored_afhps[:n_examples] + 1e-10
                )
                if np.all(np.abs(factor_check - 100) < 1):
                    print(
                        "  → Stored values appear to be fractions (0-1) instead of "
                        "percentages (0-100)"
                    )

    # Use calculated AFHPs instead of the ones from the file
    afhps = np.array(calculated_afhps)

    print(f"  AFHP range (calculated): {afhps.min():.2f}% - {afhps.max():.2f}%")

    # --- Weak baseline on the help-asked subset per CurvePoint ---------------
    # The AFHP=0 CurvePoint is, by construction, the weak agent running the
    # whole episode. Its per-episode raw_returns let us slice the weak
    # performance by any other CurvePoint's help-asked mask.
    weak_returns_full = None
    if len(calculated_afhps) > 0:
        novice_idx = int(np.argmin(calculated_afhps))
        # Treat <1% as effectively "no help" for the baseline; reject the
        # boundary if it doesn't pass the all-False level_ood_pred check.
        if calculated_afhps[novice_idx] < 1.0:
            nov_summary = meta[novice_idx]["summary"]["test"]
            nov_returns = nov_summary.get("raw_returns", None)
            nov_pred = nov_summary.get("level_ood_pred", None)
            if (
                nov_returns is not None
                and nov_pred is not None
                and not any(bool(v) for v in nov_pred)
            ):
                weak_returns_full = np.asarray(nov_returns, dtype=float)

    weak_on_help_performances = []
    for pt_meta in meta:
        summary = pt_meta["summary"]["test"]
        level_ood_pred = summary.get("level_ood_pred", [])
        if (
            weak_returns_full is None
            or len(level_ood_pred) == 0
            or len(level_ood_pred) != len(weak_returns_full)
        ):
            weak_on_help_performances.append(np.nan)
            continue
        mask = np.asarray(level_ood_pred, dtype=bool)
        if not mask.any():
            weak_on_help_performances.append(np.nan)
            continue
        weak_on_help_performances.append(float(weak_returns_full[mask].mean()))
    weak_on_help_performances = np.asarray(weak_on_help_performances, dtype=float)

    # Additional validation: Check if performance values are reasonable
    print("\n  === Performance Value Validation ===")
    print(
        f"  Original performances: min={performances.min():.2f}, max={performances.max():.2f}"
    )

    # Handle NaN values in performance_asked
    n_nan_asked = np.sum(np.isnan(performance_asked))
    if n_nan_asked > 0:
        print(
            f"  Performance asked: min={np.nanmin(performance_asked):.2f}, max={np.nanmax(performance_asked):.2f} ({n_nan_asked} NaN values)"
        )
    else:
        print(
            f"  Performance asked: min={np.min(performance_asked):.2f}, max={np.max(performance_asked):.2f}"
        )

    # Handle NaN values in strong_performances
    n_nan_strong = np.sum(np.isnan(strong_performances))
    if n_nan_strong == len(strong_performances):
        print(
            "  Strong performances: all values are NaN. This strong reval "
            "artifact cannot contribute points to the plot."
        )
    elif n_nan_strong > 0:
        print(
            f"  Strong performances: min={np.nanmin(strong_performances):.2f}, max={np.nanmax(strong_performances):.2f} ({n_nan_strong} NaN values)"
        )
    else:
        print(
            f"  Strong performances: min={np.min(strong_performances):.2f}, max={np.max(strong_performances):.2f}"
        )

    # Check if NaN positions match
    if n_nan_asked > 0 or n_nan_strong > 0:
        nan_mask_asked = np.isnan(performance_asked)
        nan_mask_strong = np.isnan(strong_performances)
        if np.array_equal(nan_mask_asked, nan_mask_strong):
            print(
                "  ✓ NaN positions match between performance_asked and "
                "strong_performances"
            )
        else:
            print("  WARNING: NaN positions don't match between arrays!")
            print(f"    NaN in performance_asked: {np.where(nan_mask_asked)[0]}")
            print(f"    NaN in strong_performances: {np.where(nan_mask_strong)[0]}")

    # Verify that strong performances are per-checkpoint averages for help-requested episodes
    if len(strong_meta) > 0:
        # Spot check a few checkpoints to ensure alignment
        n_checks = min(3, len(calculated_afhps))
        print(f"\n  === Checkpoint Alignment Spot Check (first {n_checks}) ===")
        for i in range(n_checks):
            orig_afhp = calculated_afhps[i]
            strong_perf = strong_performances[i]
            perf_asked = performance_asked[i]
            print(
                f"  Checkpoint {i}: AFHP={orig_afhp:.2f}%, Perf(asked)={perf_asked:.2f}, Strong={strong_perf:.2f}"
            )

    return (
        afhps,
        performances,
        np.array(performance_asked),
        strong_performances,
        weak_on_help_performances,
    )


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
            x_sorted, y_sorted, kind="linear", bounds_error=False, fill_value=np.nan
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

    return (
        common_x,
        np.array(y_medians),
        np.array(y_lower_quantiles),
        np.array(y_upper_quantiles),
    )


def interpolate_curves_at_x(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    x_target: float,
) -> List[float]:
    """Interpolate each per-experiment curve at x_target, dropping NaNs/out-of-range."""
    values: List[float] = []
    for x, y in zip(x_arrays, y_arrays):
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < 2:
            continue
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        sort_idx = np.argsort(x_valid)
        x_sorted = x_valid[sort_idx]
        y_sorted = y_valid[sort_idx]
        f = interpolate.interp1d(
            x_sorted, y_sorted, kind="linear", bounds_error=False, fill_value=np.nan
        )
        v = float(f(x_target))
        if not np.isnan(v):
            values.append(v)
    return values


def plot_strong_reval_diff(
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    env_filter: Optional[str],
    exp_id_filter: Optional[Set[int]] = None,
    method_order: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    no_aggregate: bool = False,
    plot_absolute: bool = False,
    plot_strong_only: bool = False,
    afhp_mark: Optional[float] = None,
    normalize_by_subset: bool = False,
    strong_vs_baseline: bool = False,
    paper_mode: bool = False,
):
    """
    Plot the performance difference between coordination policy and strong-from-start.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes
        env_filter: Only include runs for this environment
        exp_id_filter: Only include these experiment IDs
        method_order: Order of methods to plot
        method_filter: Methods to exclude
        save_path: Path to save the figure
        title: Custom title for the plot
        no_aggregate: Plot experiments separately instead of aggregating
        plot_absolute: If True, plot absolute performances instead of differences
        plot_strong_only: If True, only plot strong agent performance (not differences)
    """
    global plt, sns
    if plt is None:
        import matplotlib.pyplot as pyplot

        plt = pyplot
    if sns is None:
        import seaborn as seaborn

        sns = seaborn

    results = extract_strong_reval_results(
        eval_dir, prefix_filter, env_filter, exp_id_filter
    )

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
    if paper_mode:
        setup_plot_style(paper_mode=True, use_latex=True)
        plt.figure(figsize=(8, 4.5))
    else:
        plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(valid_methods))
    line_styles = get_line_styles(len(valid_methods), paper_mode, valid_methods)
    plotted_any = False
    mark_values_per_method: Dict[str, List[float]] = {}

    for method_idx, method in enumerate(valid_methods):
        exp_data = results[method]
        exp_ids = sorted(exp_data.keys())

        if len(exp_ids) == 0:
            print(f"Warning: No experiments found for {method}, skipping...")
            continue

        # Load all experiment data
        x_arrays = []
        y_arrays = []
        # Used only by the normalize_by_subset path so the denominator can
        # be averaged across this method's experiments per CurvePoint.
        norm_afhps_per_exp: List[np.ndarray] = []
        norm_diff_per_exp: List[np.ndarray] = []
        norm_weak_per_exp: List[np.ndarray] = []
        norm_strong_per_exp: List[np.ndarray] = []

        for exp_id in exp_ids:
            original_npz, strong_reval_npz = exp_data[exp_id]

            try:
                (
                    afhps,
                    performances,
                    performance_asked,
                    strong_performances,
                    weak_on_help_performances,
                ) = load_performance_data(original_npz, strong_reval_npz)

                # Calculate what to plot based on options
                if strong_vs_baseline:
                    # Expert on all levels = AFHP=100% endpoint
                    expert_idx = int(np.argmax(afhps))
                    if afhps[expert_idx] < 90.0:
                        print(
                            f"  Warning: max AFHP for {method} exp{exp_id} is "
                            f"{afhps[expert_idx]:.1f}% — no reliable all-levels "
                            f"expert baseline; skipping."
                        )
                    else:
                        expert_all = performances[expert_idx]
                        diff = expert_all - strong_performances
                        valid_mask = ~np.isnan(diff)
                        if np.sum(valid_mask) > 0:
                            x_arrays.append(afhps[valid_mask])
                            y_arrays.append(diff[valid_mask])
                elif plot_strong_only:
                    # Only plot strong agent performance
                    valid_mask = ~np.isnan(strong_performances)
                    if np.sum(valid_mask) > 0:
                        x_arrays.append(afhps[valid_mask])
                        y_arrays.append(strong_performances[valid_mask])
                elif plot_absolute:
                    # Plot both absolute values (coordination and strong)
                    # For now, just plot coordination performance
                    valid_mask = ~np.isnan(performance_asked)
                    if np.sum(valid_mask) > 0:
                        x_arrays.append(afhps[valid_mask])
                        y_arrays.append(performance_asked[valid_mask])
                elif normalize_by_subset:
                    # Defer normalization — we need the per-CurvePoint averages
                    # of weak/strong across this method's experiments first, so
                    # every experiment's diff is divided by the SAME shared
                    # denominator at each AFHP.
                    diff = performance_asked - strong_performances
                    valid_mask = ~(
                        np.isnan(performance_asked)
                        | np.isnan(strong_performances)
                        | np.isnan(weak_on_help_performances)
                    )
                    if np.sum(valid_mask) > 0:
                        norm_afhps_per_exp.append(afhps[valid_mask])
                        norm_diff_per_exp.append(diff[valid_mask])
                        norm_weak_per_exp.append(
                            weak_on_help_performances[valid_mask]
                        )
                        norm_strong_per_exp.append(strong_performances[valid_mask])
                else:
                    # Plot raw difference: performance_asked - strong_performances.
                    diff = performance_asked - strong_performances
                    valid_mask = ~(
                        np.isnan(performance_asked) | np.isnan(strong_performances)
                    )
                    if np.sum(valid_mask) > 0:
                        x_arrays.append(afhps[valid_mask])
                        y_arrays.append(diff[valid_mask])

            except Exception as e:
                print(f"Warning: Failed to load data for {method} exp{exp_id}: {e}")
                continue

        # If normalize_by_subset was requested, build the shared per-CurvePoint
        # denominator from this method's experiments before populating
        # x_arrays / y_arrays.
        if normalize_by_subset and norm_afhps_per_exp:
            # Common AFHP grid = union of every experiment's AFHP values.
            all_xs: set = set()
            for x in norm_afhps_per_exp:
                all_xs.update(x.tolist())
            common_x = np.array(sorted(all_xs))

            def _interp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                sort_idx = np.argsort(x)
                f = interpolate.interp1d(
                    x[sort_idx],
                    y[sort_idx],
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                return f(common_x)

            weak_grid = np.array(
                [_interp(x, w) for x, w in zip(norm_afhps_per_exp, norm_weak_per_exp)]
            )
            strong_grid = np.array(
                [
                    _interp(x, s)
                    for x, s in zip(norm_afhps_per_exp, norm_strong_per_exp)
                ]
            )
            diff_grid = np.array(
                [_interp(x, d) for x, d in zip(norm_afhps_per_exp, norm_diff_per_exp)]
            )

            with np.errstate(invalid="ignore"):
                mean_weak = np.nanmean(weak_grid, axis=0)
                mean_strong = np.nanmean(strong_grid, axis=0)
            shared_denom = mean_strong - mean_weak

            for diff_row in diff_grid:
                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_row = np.where(
                        np.abs(shared_denom) > 1e-9,
                        diff_row / shared_denom,
                        np.nan,
                    )
                valid = ~np.isnan(norm_row)
                if valid.any():
                    x_arrays.append(common_x[valid])
                    y_arrays.append(norm_row[valid])

        if len(x_arrays) == 0:
            print(f"Warning: No valid data for {method}, skipping...")
            continue

        if afhp_mark is not None:
            mark_values_per_method[method] = interpolate_curves_at_x(
                x_arrays, y_arrays, afhp_mark
            )

        # Get display name
        label = METHOD_NAMES.get(method, method)
        linestyle = line_styles[method_idx]

        if len(x_arrays) == 1 or no_aggregate:
            # Single experiment or no aggregation mode
            if no_aggregate and len(x_arrays) > 1:
                # Plot each experiment separately
                base_color = colors[method_idx]
                for i, (x, y, exp_id) in enumerate(zip(x_arrays, y_arrays, exp_ids)):
                    sort_idx = np.argsort(x)
                    alpha = 0.7 + (i / len(x_arrays)) * 0.3
                    if paper_mode:
                        exp_label = format_label(method, paper_mode)
                    else:
                        exp_label = f"{label} exp{exp_id}"

                    plt.plot(
                        x[sort_idx],
                        y[sort_idx],
                        label=exp_label,
                        color=base_color,
                        linestyle=linestyle,
                        alpha=alpha,
                        marker="o" if method == "wait" else None,
                        markersize=3 if method == "wait" else None,
                        linewidth=1.5,
                    )
                    plotted_any = True
            else:
                # Single experiment
                x, y = x_arrays[0], y_arrays[0]
                sort_idx = np.argsort(x)
                if paper_mode:
                    single_label = format_label(method, paper_mode, n_experiments=1)
                else:
                    single_label = f"{label} (n=1)"
                plt.plot(
                    x[sort_idx],
                    y[sort_idx],
                    label=single_label,
                    color=colors[method_idx],
                    linestyle=linestyle,
                    marker="o" if method == "wait" else None,
                    markersize=4,
                )
                plotted_any = True
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

            if len(common_x) == 0:
                print(f"Warning: Aggregated data for {method} is all NaN, skipping...")
                continue

            n_exps = len(x_arrays)

            if paper_mode:
                agg_label = format_label(method, paper_mode, n_experiments=n_exps)
            else:
                agg_label = f"{label} (n={n_exps})"

            # Plot median line
            plt.plot(
                common_x,
                y_median,
                label=agg_label,
                color=colors[method_idx],
                linestyle=linestyle,
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
            plotted_any = True

    if not plotted_any:
        env_str = env_filter if env_filter else "all"
        prefix_str = ",".join(prefix_filter) if prefix_filter else "all"
        print(
            "No valid non-NaN strong reval points were found for "
            f"env={env_str}, prefix={prefix_str}. Check whether the selected "
            "prefix has completed *_strong_reval.npz artifacts with help "
            "seeds."
        )
        plt.close()
        return

    # Add reference line at y=0 (only for difference plots). In paper mode the
    # legend entry is normally suppressed, but --strong-vs-baseline keeps it so
    # the "No difference" reference is explained.
    if not plot_strong_only and not plot_absolute:
        show_zero_label = strong_vs_baseline or not paper_mode
        plt.axhline(
            y=0,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="No difference" if show_zero_label else None,
        )

    # Mark a specific AFHP value, if requested.
    if afhp_mark is not None:
        plt.axvline(
            x=afhp_mark,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"AFHP={afhp_mark:g}%",
        )
        for method_idx, method in enumerate(valid_methods):
            vals = mark_values_per_method.get(method, [])
            if not vals:
                continue
            y_mark = float(np.median(vals))
            plt.plot(
                [afhp_mark],
                [y_mark],
                marker="o",
                color=colors[method_idx],
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="None",
            )
            plt.annotate(
                f"{y_mark:.2f}",
                xy=(afhp_mark, y_mark),
                xytext=(6, 0),
                textcoords="offset points",
                color=colors[method_idx],
                fontsize=9,
                va="center",
                ha="left",
            )

    # Labels and title
    env_str = env_filter if env_filter else "all"
    prefix_str = ",".join(prefix_filter) if prefix_filter else "all"

    plt.xlabel("Ask-For-Help Percentage (AFHP)")

    if strong_vs_baseline:
        plt.ylabel(r"Return $\Delta$")
        default_title = (
            "Expert Performance Gap: All Levels vs Help-Requested Subset "
            f"({env_str}, prefix={prefix_str})"
        )
    elif plot_strong_only:
        plt.ylabel("Mean Return (Strong Agent from Start)")
        default_title = f"Strong Agent Performance on Help-Requested Episodes ({env_str}, prefix={prefix_str})"
    elif plot_absolute:
        plt.ylabel("Mean Return (Coordination Policy)")
        default_title = (
            f"Coordination Policy Performance ({env_str}, prefix={prefix_str})"
        )
    else:
        if normalize_by_subset:
            plt.ylabel(r"Mean Return $\Delta$ (normalized)")
            default_title = (
                "Performance Loss from Mid-Episode Switching, normalized by "
                f"per-subset (Strong - Weak) ({env_str}, prefix={prefix_str})"
            )
        else:
            plt.ylabel(r"Mean Return $\Delta$")
            default_title = (
                "Performance Loss from Mid-Episode Switching "
                f"({env_str}, prefix={prefix_str})"
            )

    if title:
        plt.title(title)
    elif not paper_mode:
        plt.title(default_title)

    if paper_mode:
        style_plot_for_publication(
            legend_outside=True,
            legend_location="center left",
            legend_bbox_to_anchor=(1.02, 0.5),
        )
    else:
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if afhp_mark is not None:
        if strong_vs_baseline:
            metric_name = "Expert all - Expert help subset"
        elif plot_strong_only:
            metric_name = "Strong return"
        elif plot_absolute:
            metric_name = "Coordination return"
        else:
            metric_name = "Performance diff (Coordination - Strong)"
        print(f"\n=== {metric_name} at AFHP={afhp_mark:g}% ===")
        for method in valid_methods:
            vals = mark_values_per_method.get(method, [])
            label = METHOD_NAMES.get(method, method)
            if not vals:
                print(f"  {label}: no curve covers AFHP={afhp_mark:g}%")
                continue
            arr = np.array(vals)
            print(
                f"  {label}: median={np.median(arr):.3f}, "
                f"mean={arr.mean():.3f}, n={len(arr)}, "
                f"values={[round(v, 3) for v in vals]}"
            )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    elif matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            "Matplotlib is using a non-interactive Agg backend even though no "
            "--save path was provided. This means the backend was selected "
            "before plotting. Use --save PATH, or set MPLBACKEND=TkAgg and rerun."
        )
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
        help="Prefix filter(s) for experiment directories (e.g., 'dummy04' or 'dummy04 dummy05')",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["coinrun", "maze", "maze_afh", "heist"],
        help="Environment filter",
    )
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        default=None,
        help="Only include these experiment IDs (e.g. --exp-ids 0 2)",
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
    parser.add_argument(
        "--strong-only",
        action="store_true",
        help="Plot only strong agent performance (from start) vs AFHP",
    )
    parser.add_argument(
        "--afhp-mark",
        type=float,
        default=None,
        help="Mark and print the interpolated value at this AFHP (e.g. --afhp-mark 50)",
    )
    parser.add_argument(
        "--normalize-by-subset",
        "--normalize_by_subset",
        dest="normalize_by_subset",
        action="store_true",
        help=(
            "Divide the (coord - strong-from-start) difference by the "
            "(strong - weak) range computed *on each CurvePoint's help-asked "
            "subset*, turning the y-axis into the fraction of the available "
            "improvement that was lost by handing over mid-episode. Has no "
            "effect when combined with --absolute or --strong-only."
        ),
    )
    parser.add_argument(
        "--strong-vs-baseline",
        action="store_true",
        help=(
            "Plot (expert on all levels) - (expert on help-requested subset) "
            "vs AFHP. Shows whether methods select levels where the expert "
            "performs worse than its overall average."
        ),
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Paper-ready styling: LaTeX fonts, per-method linestyles, no title, "
            "legend outside the axes. Applies to all plot modes."
        ),
    )

    args = parser.parse_args()

    configure_matplotlib_backend(args.save)

    global plt, sns
    import matplotlib.pyplot as pyplot
    import seaborn as seaborn

    plt = pyplot
    sns = seaborn

    eval_dir = Path(args.eval_dir)

    # Parse method order
    method_order = None
    if args.method_order:
        method_order = [m.strip() for m in args.method_order.split(",")]

    plot_strong_reval_diff(
        eval_dir=eval_dir,
        prefix_filter=args.prefix,
        env_filter=args.env,
        exp_id_filter=set(args.exp_ids) if args.exp_ids is not None else None,
        method_order=method_order,
        method_filter=args.method_filter,
        save_path=args.save,
        title=args.title,
        no_aggregate=args.no_aggregate,
        plot_absolute=args.absolute,
        plot_strong_only=args.strong_only,
        afhp_mark=args.afhp_mark,
        normalize_by_subset=args.normalize_by_subset,
        strong_vs_baseline=args.strong_vs_baseline,
        paper_mode=args.paper,
    )


if __name__ == "__main__":
    main()

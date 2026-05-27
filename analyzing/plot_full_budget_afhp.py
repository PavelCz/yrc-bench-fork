#!/usr/bin/env python3
"""Plot AFHP changes from full-budget timeout-reset evaluation."""

from __future__ import annotations

import argparse
import math
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


def configure_matplotlib_backend(save_path: Optional[str]) -> None:
    if save_path:
        matplotlib.use("Agg", force=True)
        return

    try:
        matplotlib.use("TkAgg", force=True)
    except ImportError as exc:
        raise RuntimeError(
            "Could not load Matplotlib's TkAgg backend for interactive display. "
            "Use --save PATH for a non-interactive run."
        ) from exc


def ensure_plotting_imports():
    global plt, sns
    if plt is None:
        import matplotlib.pyplot as pyplot

        plt = pyplot
    if sns is None:
        import seaborn as seaborn

        sns = seaborn


def parse_experiment_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    match = re.match(r"^(.+)_(coinrun|maze|maze_afh|heist)_exp(\d+)$", dir_name)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None


def parse_method_dir(dir_name: str) -> Optional[Tuple[str, str, int]]:
    match = re.match(r"^(coinrun|maze|maze_afh|heist)_(.+)_exp(\d+)$", dir_name)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None


def extract_full_budget_results(
    eval_dir: Path,
    prefix_filter: Optional[List[str]] = None,
    env_filter: Optional[str] = None,
    exp_id_filter: Optional[Set[int]] = None,
) -> Dict[str, Dict[int, Tuple[Path, Path]]]:
    results: Dict[str, Dict[int, Tuple[Path, Path]]] = defaultdict(dict)

    for child in eval_dir.iterdir():
        if not child.is_dir():
            continue

        parsed = parse_experiment_dir(child.name)
        if parsed is None:
            continue

        prefix, env, exp_id = parsed
        if prefix_filter is not None and prefix not in prefix_filter:
            continue
        if env_filter is not None and env != env_filter:
            continue
        if exp_id_filter is not None and exp_id not in exp_id_filter:
            continue

        for method_dir in child.iterdir():
            if not method_dir.is_dir():
                continue

            parsed_method = parse_method_dir(method_dir.name)
            if parsed_method is None:
                method_name = method_dir.name
            else:
                _, method_name, method_exp_id = parsed_method
                if method_exp_id != exp_id:
                    continue

            original_npz = None
            full_budget_npz = None
            for run_dir in method_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                run_original_npz = None
                run_full_budget_npz = None
                for npz_file in run_dir.glob("*.npz"):
                    if npz_file.name.endswith("_full_budget_eval.npz"):
                        run_full_budget_npz = npz_file
                    elif npz_file.name.startswith(
                        "eval_seed_"
                    ) and npz_file.name.endswith("_test.npz"):
                        run_original_npz = npz_file

                if run_original_npz is None or run_full_budget_npz is None:
                    continue

                if (
                    original_npz is None
                    or run_dir.stat().st_mtime > original_npz.parent.stat().st_mtime
                ):
                    original_npz = run_original_npz
                    full_budget_npz = run_full_budget_npz

            if original_npz is not None and full_budget_npz is not None:
                results[method_name][exp_id] = (original_npz, full_budget_npz)

    return dict(results)


def normalize_afhp_percent(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) > 0 and np.nanmax(np.abs(finite)) <= 1.5:
        return values * 100.0
    return values


def mean_or_nan(values) -> float:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def help_only_performances_from_original(original_npz: Path) -> np.ndarray:
    data = np.load(original_npz, allow_pickle=True)
    help_performances = []
    for point_meta in data["meta"]:
        summary = point_meta["summary"]["test"]
        level_ood_pred = summary.get("level_ood_pred", [])
        raw_returns = summary.get("raw_returns", [])
        help_returns = [
            ret for pred, ret in zip(level_ood_pred, raw_returns) if bool(pred)
        ]
        help_performances.append(mean_or_nan(help_returns))
    return np.array(help_performances, dtype=float)


def help_only_performances_from_full_budget(full_budget_meta) -> np.ndarray:
    help_performances = []
    for point_meta in full_budget_meta:
        summary = point_meta.get("summary", {})
        level_ood_pred = summary.get("level_ood_pred", [])
        raw_returns = summary.get("raw_returns", [])
        help_returns = [
            ret for pred, ret in zip(level_ood_pred, raw_returns) if bool(pred)
        ]
        help_performances.append(mean_or_nan(help_returns))
    return np.array(help_performances, dtype=float)


def load_full_budget_data(full_budget_npz: Path, original_npz: Optional[Path] = None):
    data = np.load(full_budget_npz, allow_pickle=True)
    required = {
        "thresholds",
        "original_afhps",
        "full_budget_afhps",
        "original_performances",
        "full_budget_performances",
        "full_budget_meta",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(
            f"{full_budget_npz} missing required fields: {sorted(missing)}"
        )

    thresholds = np.asarray(data["thresholds"], dtype=float)
    original_afhps = normalize_afhp_percent(data["original_afhps"])
    full_budget_afhps = normalize_afhp_percent(data["full_budget_afhps"])
    original_performances = np.asarray(data["original_performances"], dtype=float)
    full_budget_performances = np.asarray(data["full_budget_performances"], dtype=float)
    full_budget_meta = data["full_budget_meta"]

    lengths = {
        len(thresholds),
        len(original_afhps),
        len(full_budget_afhps),
        len(original_performances),
        len(full_budget_performances),
        len(full_budget_meta),
    }
    if len(lengths) != 1:
        raise ValueError(f"{full_budget_npz} has misaligned output arrays")

    original_help_performances = np.full(len(thresholds), np.nan, dtype=float)
    if original_npz is not None:
        original_help_performances = help_only_performances_from_original(original_npz)
        if len(original_help_performances) != len(thresholds):
            raise ValueError(
                f"{original_npz} help-only performances do not align with "
                f"{full_budget_npz}"
            )

    full_budget_help_performances = help_only_performances_from_full_budget(
        full_budget_meta
    )

    run_metadata = data["run_metadata"].item() if "run_metadata" in data.files else {}
    return {
        "thresholds": thresholds,
        "original_afhps": original_afhps,
        "full_budget_afhps": full_budget_afhps,
        "afhp_diff": full_budget_afhps - original_afhps,
        "original_performances": original_performances,
        "full_budget_performances": full_budget_performances,
        "original_help_performances": original_help_performances,
        "full_budget_help_performances": full_budget_help_performances,
        "run_metadata": run_metadata,
    }


def calculate_quantile_bands(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    quantiles: Tuple[float, float] = (0.25, 0.75),
):
    all_x_values = set()
    for x_arr in x_arrays:
        all_x_values.update(np.asarray(x_arr, dtype=float).tolist())
    common_x = np.array(sorted(all_x_values))

    interp_funcs = []
    for x, y in zip(x_arrays, y_arrays):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 2:
            continue

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        sort_idx = np.argsort(x_valid)
        x_sorted = x_valid[sort_idx]
        y_sorted = y_valid[sort_idx]
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        unique_y = y_sorted[unique_indices]
        if len(unique_x) < 2:
            continue

        interp_funcs.append(
            interpolate.interp1d(
                unique_x,
                unique_y,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        )

    y_medians = []
    y_lower = []
    y_upper = []
    for x_val in common_x:
        y_values = [f(x_val) for f in interp_funcs]
        y_values = np.array([v for v in y_values if not np.isnan(v)], dtype=float)
        if len(y_values) == 0:
            y_medians.append(np.nan)
            y_lower.append(np.nan)
            y_upper.append(np.nan)
        else:
            y_medians.append(np.median(y_values))
            y_lower.append(np.quantile(y_values, quantiles[0]))
            y_upper.append(np.quantile(y_values, quantiles[1]))

    return common_x, np.array(y_medians), np.array(y_lower), np.array(y_upper)


def sorted_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    return x[order], y[order]


def original_endpoint_line(x, y):
    x_sorted, y_sorted = sorted_xy(x, y)
    if len(x_sorted) < 2:
        return None

    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]
    if len(unique_x) < 2 or unique_x[0] > 0 or unique_x[-1] < 100:
        return None

    y_at_zero = float(np.interp(0.0, unique_x, unique_y))
    y_at_hundred = float(np.interp(100.0, unique_x, unique_y))
    return np.array([0.0, 100.0]), np.array([y_at_zero, y_at_hundred])


def weak_expert_performances(afhps, performances):
    x_sorted, y_sorted = sorted_xy(afhps, performances)
    if len(x_sorted) < 2:
        return None, None
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]
    if len(unique_x) < 2:
        return None, None
    weak = float(np.interp(0.0, unique_x, unique_y))
    expert = float(np.interp(100.0, unique_x, unique_y))
    return weak, expert


def plot_perf_diff_mode(
    results,
    valid_methods,
    *,
    no_aggregate: bool,
    help_only: bool,
    paper_mode: bool = False,
):
    colors = sns.color_palette("husl", len(valid_methods))
    line_styles = get_line_styles(len(valid_methods), paper_mode, valid_methods)
    plotted_any = False

    for method_idx, method in enumerate(valid_methods):
        x_arrays = []
        y_arrays = []
        exp_ids = sorted(results[method].keys())

        for exp_id in exp_ids:
            original_npz, full_budget_npz = results[method][exp_id]
            try:
                data = load_full_budget_data(full_budget_npz, original_npz)
            except Exception as exc:
                print(f"Warning: failed to load {method} exp{exp_id}: {exc}")
                continue

            if help_only:
                original_perf = data["original_help_performances"]
                full_perf = data["full_budget_help_performances"]
            else:
                original_perf = data["original_performances"]
                full_perf = data["full_budget_performances"]

            weak, expert = weak_expert_performances(
                data["original_afhps"], data["original_performances"]
            )
            if weak is None or expert is None or not np.isfinite(expert - weak) or (expert - weak) == 0:
                print(
                    f"Warning: cannot normalize {method} exp{exp_id} "
                    f"(weak={weak}, expert={expert}); skipping."
                )
                continue

            normalized_diff = (full_perf - original_perf) / (expert - weak)
            x_arrays.append(data["original_afhps"])
            y_arrays.append(normalized_diff)

        if len(x_arrays) == 0:
            print(f"Warning: no valid perf-diff data for {method}, skipping")
            continue

        base_color = colors[method_idx]
        linestyle = line_styles[method_idx]
        if len(x_arrays) == 1 or no_aggregate:
            for i, (x, y) in enumerate(zip(x_arrays, y_arrays)):
                x_sorted, y_sorted = sorted_xy(x, y)
                if len(x_sorted) == 0:
                    continue
                if len(x_arrays) == 1:
                    exp_label = format_label(method, paper_mode, n_experiments=1)
                elif paper_mode:
                    exp_label = format_label(method, paper_mode)
                else:
                    base = format_label(method, paper_mode)
                    exp_label = f"{base} exp{exp_ids[i]}"
                plt.plot(
                    x_sorted,
                    y_sorted,
                    label=exp_label,
                    color=base_color,
                    linestyle=linestyle,
                    alpha=0.8,
                    linewidth=1.8,
                )
                plotted_any = True
        else:
            common_x, y_median, y_lower, y_upper = calculate_quantile_bands(
                x_arrays, y_arrays
            )
            valid = ~np.isnan(y_median)
            if np.sum(valid) == 0:
                continue
            plt.plot(
                common_x[valid],
                y_median[valid],
                label=format_label(method, paper_mode, n_experiments=len(x_arrays)),
                color=base_color,
                linestyle=linestyle,
                linewidth=2,
            )
            plt.fill_between(
                common_x[valid],
                y_lower[valid],
                y_upper[valid],
                color=base_color,
                alpha=0.2,
            )
            plotted_any = True

    if not plotted_any:
        print("No valid non-NaN perf-diff points were found.")
        return False

    zero_line_label = None if paper_mode else "No change"
    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=zero_line_label,
    )
    if paper_mode:
        plt.xlabel(r"Ask-For-Help Percentage (AFHP)")
        plt.ylabel(r"Return $\Delta$ (normalized)")
    else:
        plt.xlabel("Regular Eval AFHP (%)")
        perf_scope = "Help-Only Return" if help_only else "Return"
        plt.ylabel(
            f"(Full-Budget {perf_scope} - Regular {perf_scope}) / (Expert - Weak)"
        )
    return True


def plot_diff_mode(
    results,
    valid_methods,
    *,
    no_aggregate: bool,
):
    colors = sns.color_palette("husl", len(valid_methods))
    plotted_any = False

    for method_idx, method in enumerate(valid_methods):
        x_arrays = []
        y_arrays = []
        exp_ids = sorted(results[method].keys())

        for exp_id in exp_ids:
            _, full_budget_npz = results[method][exp_id]
            try:
                data = load_full_budget_data(full_budget_npz)
            except Exception as exc:
                print(f"Warning: failed to load {method} exp{exp_id}: {exc}")
                continue
            x_arrays.append(data["original_afhps"])
            y_arrays.append(data["afhp_diff"])

        if len(x_arrays) == 0:
            print(f"Warning: no valid full-budget data for {method}, skipping")
            continue

        label = METHOD_NAMES.get(method, method)
        if len(x_arrays) == 1 or no_aggregate:
            base_color = colors[method_idx]
            for i, (x, y) in enumerate(zip(x_arrays, y_arrays)):
                x_sorted, y_sorted = sorted_xy(x, y)
                if len(x_sorted) == 0:
                    continue
                exp_label = label if len(x_arrays) == 1 else f"{label} exp{exp_ids[i]}"
                plt.plot(
                    x_sorted,
                    y_sorted,
                    label=exp_label,
                    color=base_color,
                    alpha=0.8,
                    linewidth=1.8,
                )
                plotted_any = True
        else:
            common_x, y_median, y_lower, y_upper = calculate_quantile_bands(
                x_arrays, y_arrays
            )
            valid = ~np.isnan(y_median)
            if np.sum(valid) == 0:
                continue
            plt.plot(
                common_x[valid],
                y_median[valid],
                label=f"{label} (n={len(x_arrays)})",
                color=colors[method_idx],
                linewidth=2,
            )
            plt.fill_between(
                common_x[valid],
                y_lower[valid],
                y_upper[valid],
                color=colors[method_idx],
                alpha=0.2,
            )
            plotted_any = True

    if not plotted_any:
        print("No valid non-NaN full-budget AFHP points were found.")
        return False

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="No change")
    plt.xlabel("Regular Eval AFHP (%)")
    plt.ylabel("Full-Budget AFHP - Regular AFHP (percentage points)")
    return True


def plot_overlay_mode(
    results,
    valid_methods,
    *,
    no_aggregate: bool,
    help_only: bool,
):
    n_methods = len(valid_methods)
    n_cols = 2 if n_methods == 4 else min(3, n_methods)
    n_rows = int(math.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    colors = sns.color_palette("deep", 2)
    plotted_any = False

    for method_idx, method in enumerate(valid_methods):
        ax = axes[method_idx // n_cols][method_idx % n_cols]
        exp_ids = sorted(results[method].keys())
        label = METHOD_NAMES.get(method, method)
        ax.set_title(label)

        original_x_arrays = []
        original_y_arrays = []
        full_x_arrays = []
        full_y_arrays = []

        for exp_id in exp_ids:
            original_npz, full_budget_npz = results[method][exp_id]
            try:
                data = load_full_budget_data(full_budget_npz, original_npz)
            except Exception as exc:
                print(f"Warning: failed to load {method} exp{exp_id}: {exc}")
                continue

            original_x_arrays.append(data["original_afhps"])
            full_x_arrays.append(data["full_budget_afhps"])
            if help_only:
                original_y_arrays.append(data["original_help_performances"])
                full_y_arrays.append(data["full_budget_help_performances"])
            else:
                original_y_arrays.append(data["original_performances"])
                full_y_arrays.append(data["full_budget_performances"])

        if len(original_x_arrays) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        if len(original_x_arrays) == 1 or no_aggregate:
            for i, exp_id in enumerate(exp_ids[: len(original_x_arrays)]):
                orig_x, orig_y = sorted_xy(original_x_arrays[i], original_y_arrays[i])
                full_x, full_y = sorted_xy(full_x_arrays[i], full_y_arrays[i])
                suffix = "" if len(original_x_arrays) == 1 else f" exp{exp_id}"
                ax.plot(
                    orig_x,
                    orig_y,
                    color=colors[0],
                    linestyle="--",
                    alpha=0.8,
                    label=f"Regular{suffix}",
                )
                endpoint_line = original_endpoint_line(orig_x, orig_y)
                if endpoint_line is not None:
                    line_x, line_y = endpoint_line
                    ax.plot(
                        line_x,
                        line_y,
                        color="black",
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.7,
                        label=f"Original endpoint line{suffix}",
                    )
                ax.plot(
                    full_x,
                    full_y,
                    color=colors[1],
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"Full budget{suffix}",
                )
                plotted_any = True
        else:
            orig_x, orig_median, orig_lower, orig_upper = calculate_quantile_bands(
                original_x_arrays, original_y_arrays
            )
            full_x, full_median, full_lower, full_upper = calculate_quantile_bands(
                full_x_arrays, full_y_arrays
            )
            orig_valid = ~np.isnan(orig_median)
            full_valid = ~np.isnan(full_median)
            ax.plot(
                orig_x[orig_valid],
                orig_median[orig_valid],
                color=colors[0],
                linestyle="--",
                label=f"Regular (n={len(original_x_arrays)})",
            )
            endpoint_line = original_endpoint_line(
                orig_x[orig_valid], orig_median[orig_valid]
            )
            if endpoint_line is not None:
                line_x, line_y = endpoint_line
                ax.plot(
                    line_x,
                    line_y,
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.7,
                    label="Original endpoint line",
                )
            ax.fill_between(
                orig_x[orig_valid],
                orig_lower[orig_valid],
                orig_upper[orig_valid],
                color=colors[0],
                alpha=0.15,
            )
            ax.plot(
                full_x[full_valid],
                full_median[full_valid],
                color=colors[1],
                linewidth=1.8,
                label=f"Full budget (n={len(full_x_arrays)})",
            )
            ax.fill_between(
                full_x[full_valid],
                full_lower[full_valid],
                full_upper[full_valid],
                color=colors[1],
                alpha=0.15,
            )
            plotted_any = True

        ax.set_xlabel("AFHP (%)")
        if help_only:
            ax.set_ylabel("Mean Return on Help-Requested Episodes")
        else:
            ax.set_ylabel("Mean Return")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for idx in range(n_methods, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    return plotted_any


def plot_full_budget_afhp(
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    env_filter: Optional[str],
    exp_id_filter: Optional[Set[int]] = None,
    method_order: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    no_aggregate: bool = False,
    overlay: bool = False,
    help_only: bool = False,
    perf_diff: bool = False,
    paper_mode: bool = False,
):
    ensure_plotting_imports()

    if paper_mode and not perf_diff:
        print("Warning: --paper currently only affects --perf-diff mode.")

    results = extract_full_budget_results(
        eval_dir, prefix_filter, env_filter, exp_id_filter
    )

    if not results:
        print("No results found with both original and full-budget eval files.")
        return

    if method_order is None:
        method_order = sorted(results.keys())
    if method_filter is not None:
        method_order = [m for m in method_order if m not in method_filter]
    valid_methods = [m for m in method_order if m in results]
    if not valid_methods:
        print("No valid methods found to plot.")
        return

    if overlay and perf_diff:
        print("Warning: --perf-diff overrides --overlay; plotting normalized difference.")

    if help_only and not (overlay or perf_diff):
        print("Warning: --help-only only affects --overlay or --perf-diff modes.")

    is_subplot_mode = overlay and not perf_diff

    if perf_diff:
        if paper_mode:
            setup_plot_style(paper_mode=True, use_latex=True)
            plt.figure(figsize=(8, 4.5))
        else:
            plt.figure(figsize=(10, 8))
        plotted_any = plot_perf_diff_mode(
            results,
            valid_methods,
            no_aggregate=no_aggregate,
            help_only=help_only,
            paper_mode=paper_mode,
        )
    elif overlay:
        plotted_any = plot_overlay_mode(
            results,
            valid_methods,
            no_aggregate=no_aggregate,
            help_only=help_only,
        )
    else:
        plt.figure(figsize=(10, 8))
        plotted_any = plot_diff_mode(
            results,
            valid_methods,
            no_aggregate=no_aggregate,
        )

    if not plotted_any:
        plt.close()
        return

    env_str = env_filter if env_filter else "all"
    prefix_str = ",".join(prefix_filter) if prefix_filter else "all"
    skip_title = perf_diff and paper_mode and title is None
    if title:
        plt.suptitle(title) if is_subplot_mode else plt.title(title)
    elif skip_title:
        pass
    elif is_subplot_mode:
        performance_scope = (
            "Help-Only Performance Curves" if help_only else "Performance Curves"
        )
        plt.suptitle(
            f"Regular vs Full-Budget {performance_scope} "
            f"({env_str}, prefix={prefix_str})"
        )
    elif perf_diff:
        perf_scope = "Help-Only Return" if help_only else "Return"
        plt.title(
            f"Normalized {perf_scope} Change Under Full-Budget Eval "
            f"({env_str}, prefix={prefix_str})"
        )
    else:
        plt.title(
            f"AFHP Change Under Full-Budget Eval ({env_str}, prefix={prefix_str})"
        )

    if perf_diff and paper_mode:
        style_plot_for_publication(
            legend_outside=True,
            legend_location="center left",
            legend_bbox_to_anchor=(1.02, 0.5),
        )
    elif not is_subplot_mode:
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    elif matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            "Matplotlib is using the non-interactive Agg backend. Use --save PATH."
        )
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot AFHP differences from full-budget eval artifacts."
    )
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, nargs="+", default=None)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["coinrun", "maze", "maze_afh", "heist"],
    )
    parser.add_argument("--exp-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--method_order",
        "-m",
        type=str,
        default=None,
        help="Comma-separated list of methods to plot in order.",
    )
    parser.add_argument(
        "--method_filter",
        "-f",
        type=str,
        nargs="+",
        default=None,
        help="Methods to exclude from plot.",
    )
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Plot experiments separately instead of aggregating.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help=(
            "Plot regular and full-budget performance curves together, one subplot "
            "per method, using each curve's achieved AFHP on the x-axis."
        ),
    )
    parser.add_argument(
        "--help-only",
        action="store_true",
        help=(
            "In --overlay or --perf-diff mode, plot mean return only on episodes "
            "where that eval asked for help. Regular and full-budget curves use "
            "their own help-requested episode sets."
        ),
    )
    parser.add_argument(
        "--perf-diff",
        action="store_true",
        help=(
            "Plot the normalized performance difference (full-budget - regular) / "
            "(expert - weak) vs regular AFHP, instead of overlaying both curves. "
            "Expert and weak performances are taken from the regular curve's "
            "AFHP=100%% and AFHP=0%% endpoints."
        ),
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Paper-ready styling for --perf-diff: LaTeX fonts, per-method "
            "linestyles, no title, legend outside the axes. Currently only "
            "affects --perf-diff mode."
        ),
    )
    args = parser.parse_args()

    configure_matplotlib_backend(args.save)
    ensure_plotting_imports()

    method_order = None
    if args.method_order:
        method_order = [m.strip() for m in args.method_order.split(",")]

    plot_full_budget_afhp(
        eval_dir=Path(args.eval_dir),
        prefix_filter=args.prefix,
        env_filter=args.env,
        exp_id_filter=set(args.exp_ids) if args.exp_ids is not None else None,
        method_order=method_order,
        method_filter=args.method_filter,
        save_path=args.save,
        title=args.title,
        no_aggregate=args.no_aggregate,
        overlay=args.overlay,
        help_only=args.help_only,
        perf_diff=args.perf_diff,
        paper_mode=args.paper,
    )


if __name__ == "__main__":
    main()

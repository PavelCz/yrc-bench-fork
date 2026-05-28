#!/usr/bin/env python3
"""
Script to plot episode length distributions for ID vs OOD levels.

Uses similar data loading infrastructure to icml_plot.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analyzing.icml_plot import extract_icml_results
from analyzing.plotting_common import (
    setup_plot_style,
    style_plot_for_publication,
)

matplotlib.use("TkAgg")


def extract_episode_lengths_by_ood_status(
    results: Dict[str, Dict[int, Path]],
    method: str,
    threshold_idx: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """
    Extract episode lengths separated by OOD ground truth status.

    Args:
        results: Dictionary from extract_icml_results
        method: Method name to extract data for
        threshold_idx: If provided, use specific threshold index.
                      If None, uses all thresholds.

    Returns:
        Tuple of (id_lengths, ood_lengths)
    """
    id_lengths = []
    ood_lengths = []

    if method not in results:
        print(f"Warning: Method {method} not found in results")
        return id_lengths, ood_lengths

    exp_data = results[method]

    for exp_id, data_path in exp_data.items():
        try:
            eval_data = np.load(data_path, allow_pickle=True)
            meta = eval_data["meta"]

            # Determine which threshold indices to use
            if threshold_idx is not None:
                indices = [threshold_idx] if threshold_idx < len(meta) else []
            else:
                # Use first threshold by default (AFHP = 0, weak agent only)
                indices = [0]

            for idx in indices:
                if idx >= len(meta):
                    continue

                element = meta[idx]
                test_summary = element["summary"]["test"]

                episode_lengths = np.array(test_summary["episode_lengths"])
                level_ood_gt = np.array(test_summary["level_ood_gt"])

                # Separate by OOD status
                id_mask = level_ood_gt == 0
                ood_mask = level_ood_gt == 1

                id_lengths.extend(episode_lengths[id_mask].tolist())
                ood_lengths.extend(episode_lengths[ood_mask].tolist())

        except Exception as e:
            print(f"Warning: Failed to load {data_path}: {e}")
            continue

    return id_lengths, ood_lengths


def plot_episode_length_distribution(
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    env_filter: Optional[str],
    method: Optional[str] = None,
    threshold_idx: Optional[int] = None,
    bins: int = 30,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    paper_mode: bool = False,
):
    """
    Plot episode length distribution for ID vs OOD levels.

    Args:
        eval_dir: Directory containing evaluation results
        prefix_filter: Only include runs with these prefixes (list)
        env_filter: Only include runs for this environment
        method: Specific method to plot (if None, uses first available)
        threshold_idx: Specific threshold index to use
        bins: Number of histogram bins
        save_path: Path to save the figure
        title: Custom title for the plot
        paper_mode: If True, use paper-ready styling
    """
    results = extract_icml_results(eval_dir, prefix_filter, env_filter)

    if not results:
        print("No results found matching the filters.")
        return

    # Select method
    if method is None:
        method = sorted(results.keys())[0]
        print(f"No method specified, using: {method}")

    if method not in results:
        print(f"Method {method} not found. Available: {sorted(results.keys())}")
        return

    # Extract episode lengths
    id_lengths, ood_lengths = extract_episode_lengths_by_ood_status(
        results, method, threshold_idx
    )

    if not id_lengths and not ood_lengths:
        print("No episode length data found.")
        return

    # Print statistics to CLI
    print("\nEpisode Length Statistics:")
    print("-" * 50)
    print(f"ID episodes: {len(id_lengths)}")
    if id_lengths:
        print(f"  Mean length: {np.mean(id_lengths):.2f}")
        print(f"  Std dev: {np.std(id_lengths):.2f}")
        print(f"  Median: {np.median(id_lengths):.2f}")
        print(f"  Min/Max: {min(id_lengths):.0f}/{max(id_lengths):.0f}")
    print(f"\nOOD episodes: {len(ood_lengths)}")
    if ood_lengths:
        print(f"  Mean length: {np.mean(ood_lengths):.2f}")
        print(f"  Std dev: {np.std(ood_lengths):.2f}")
        print(f"  Median: {np.median(ood_lengths):.2f}")
        print(f"  Min/Max: {min(ood_lengths):.0f}/{max(ood_lengths):.0f}")
    print("-" * 50)

    # Set up plot style
    setup_plot_style(paper_mode=paper_mode, use_latex=False)

    # Create plot
    plt.rcParams["hatch.linewidth"] = 1.5
    plt.rcParams["hatch.color"] = "black"
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot histograms
    if id_lengths:
        if paper_mode:
            id_label = "ID Levels"
        else:
            id_label = (
                f"ID Levels (n={len(id_lengths)}, mean={np.mean(id_lengths):.1f})"
            )

        ax.hist(
            id_lengths,
            bins=bins,
            alpha=0.8,
            label=id_label,
            color="cornflowerblue",
            density=True,
            edgecolor="black",
            linewidth=0.5,
            hatch="////",
        )

    if ood_lengths:
        if paper_mode:
            ood_label = "OOD Levels"
        else:
            ood_label = (
                f"OOD Levels (n={len(ood_lengths)}, mean={np.mean(ood_lengths):.1f})"
            )

        ax.hist(
            ood_lengths,
            bins=bins,
            alpha=0.6,
            label=ood_label,
            color="red",
            density=True,
            edgecolor="black",
            linewidth=0.5,
            hatch="",
        )

    # Labels and title
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Density")

    env_str = env_filter if env_filter else "all"

    if title:
        ax.set_title(title)
    elif not paper_mode:
        ax.set_title(f"Episode Length Distribution ({env_str})")

    # Apply publication styling
    style_plot_for_publication(
        legend_outside=False,  # Keep legend inside for histograms
        legend_location="best",
    )

    fig.subplots_adjust(left=0.28, right=0.95, top=0.95, bottom=0.18)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def list_available_methods(
    eval_dir: Path, prefix_filter: Optional[List[str]], env_filter: Optional[str]
):
    """List available methods."""
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
        description="Plot episode length distribution for ID vs OOD levels"
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
        help="Prefix filter for experiment directories (e.g., 'icml04')",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["coinrun", "maze", "maze_afh", "heist"],
        help="Environment filter",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default=None,
        help="Method to plot (if not specified, uses first available)",
    )
    parser.add_argument(
        "--threshold_idx",
        "-t",
        type=int,
        default=None,
        help="Threshold index to use (if not specified, uses index 0 = AFHP 0%%)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30)",
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
        "--list",
        action="store_true",
        help="List available methods and exit",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use paper-ready styling (no titles, cleaner appearance)",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Handle prefix filter as list
    prefix_filter = [args.prefix] if args.prefix else None

    if args.list:
        list_available_methods(eval_dir, prefix_filter, args.env)
        return

    plot_episode_length_distribution(
        eval_dir=eval_dir,
        prefix_filter=prefix_filter,
        env_filter=args.env,
        method=args.method,
        threshold_idx=args.threshold_idx,
        bins=args.bins,
        save_path=args.save,
        title=args.title,
        paper_mode=args.paper,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to plot OOD detection metrics (AFHP, performance, etc.) from evaluation results.
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore

from analyzing.utils import extract_results, extract_x_and_y_values


import matplotlib

matplotlib.use("TkAgg")

def plot_afhp(
    name_order: Optional[List[str]],
    eval_dir: Path,
    prefix_filter: Optional[List[str]],
    x_data_key: str,
    y_data_key: str,
    disable_horizontal_lines: bool = False,
    key_filter: Optional[List[str]] = None,
    ablation_key: Optional[List[str]] = None,
    separate_figures: bool = False,
    min_timestamp: Optional[str] = None,
):
    """
    Plot AFHP (Ask for Help Percentage) vs performance.

    Args:
        name_order: List of method names to plot in order. If None, uses all available
        names.
        ablation_key: Config key(s) to differentiate multiple runs from the same method.
        separate_figures: If True, plot each curve in a separate figure in a grid
        layout.
        min_timestamp: Minimum timestamp to filter runs. Only runs with timestamps >=
        this value will be included.
    """
    results = extract_results(eval_dir, prefix_filter, ablation_key, min_timestamp)

    # Collect first and last performance values for all curves
    first_performances = []
    last_performances = []

    name_map = {
        "latent-svdd": "Latent SVDD",
        "random": "Timestep Random",
        "patient-ae": "Autoencoder",
        "center-focused": "Center-focused AE",
        "deep-svdd": "DeepSVDD",
        "oc-random": "Level-Based Random",
    }

    # If name_order is None, use all available names
    if name_order is None:
        name_order = list(results.keys())

    if key_filter is not None:
        name_order = [name for name in name_order if name not in key_filter]

    # Filter out methods that don't exist in results
    valid_names = []
    for name in name_order:
        if name not in results:
            print(f"Warning: {name} not found in evals, skipping...")
            continue
        valid_names.append(name)

    if not valid_names:
        print("No valid methods found to plot.")
        return

    if separate_figures:
        # Create a grid layout for separate figures
        n_plots = len(valid_names)
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, name in enumerate(valid_names):
            ax = axes[idx]
            data_path = results[name]

            eval_data = np.load(data_path, allow_pickle=True)
            x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

            # Store first and last performance values
            first_performances.append(y[0])
            last_performances.append(y[-1])

            if name in name_map:
                label = name_map[name]
            else:
                label = name

            sns.lineplot(x=x, y=y, ax=ax, marker="o")
            ax.set_title(f"{label}")
            ax.set_xlabel(x_data_key)
            ax.set_ylabel(y_data_key)

            # Add horizontal lines for this individual plot
            if not disable_horizontal_lines:
                ax.axhline(
                    y=y[0],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Weak Agent",
                )
                ax.axhline(
                    y=y[-1],
                    color="blue",
                    linestyle="--",
                    alpha=0.7,
                    label="Oracle",
                )
                ax.legend()

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()
    else:
        # Original behavior: all curves in one figure
        # Clear previous plot
        plt.clf()

        for name in valid_names:
            data_path = results[name]

            eval_data = np.load(data_path, allow_pickle=True)
            x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

            # Store first and last performance values
            first_performances.append(y[0])
            last_performances.append(y[-1])

            if name in name_map:
                label = name_map[name]
            else:
                label = name

            sns.lineplot(x=x, y=y, label=label, marker="o")

        # Calculate means
        mean_first_performance = np.mean(first_performances)
        mean_last_performance = np.mean(last_performances)

        # Add horizontal lines
        if not disable_horizontal_lines:
            plt.axhline(
                y=mean_first_performance,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Weak Agent",
            )
            plt.axhline(
                y=mean_last_performance,
                color="blue",
                linestyle="--",
                alpha=0.7,
                label="Oracle",
            )

        plt.xlabel(x_data_key)
        plt.ylabel(y_data_key)
        plt.title(f"{y_data_key} over {x_data_key}")
        plt.legend()
        # plt.savefig("afhp_plot.png", dpi=300, bbox_inches="tight")
        plt.show()


def eval_result_plotter():
    """Main function for OOD evaluation result plotting."""
    parser = argparse.ArgumentParser(
        description="Plot AFHP (Ask for Help Percentage) vs performance"
    )
    parser.add_argument(
        "--name_order",
        "-n",
        default=None,
        type=str,
        help=(
            "Comma-separated list of method names to plot in order. If not specified, "
            "uses all available names.",
        ),
    )  # type: ignore[arg-type]

    parser.add_argument(
        "--eval_dir",
        type=str,
        help="Directory containing the evaluation files.",
    )

    parser.add_argument(
        "--prefix_filter",
        default=None,
        type=str,
        nargs="+",
        help=(
            "Prefix filter(s) for the evaluation files. Can specify one or more "
            "prefixes to combine runs from multiple groups."
        ),
    )

    parser.add_argument(
        "--x_data_key",
        type=str,
        help="Key for the x data.",
    )

    parser.add_argument(
        "--y_data_key",
        type=str,
        help="Key for the y data.",
    )

    parser.add_argument(
        "--disable_horizontal_lines",
        action="store_true",
        help="Disable horizontal lines.",
    )

    parser.add_argument(
        "--key_filter",
        "-f",
        default=None,
        type=str,
        nargs="+",
        help="Filter out these keys.",
    )

    parser.add_argument(
        "--ablation_key",
        default=None,
        type=str,
        nargs="+",
        help=(
            "Config key(s) (with dots for nested keys) to differentiate multiple runs "
            "from the same method. Can specify multiple keys to use all of them. "
            "E.g., 'evaluation.coverage_fraction' or 'general.seed' or both"
        ),
    )

    parser.add_argument(
        "--separate_figures",
        action="store_true",
        help="Show each curve in a separate subplot in a grid layout instead of all in one figure.",
    )

    parser.add_argument(
        "--min_timestamp",
        default=None,
        type=str,
        help=(
            "Minimum timestamp to filter runs. Only runs with timestamps >= this value "
            "will be included. Format: YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS"
        ),
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    x_data_key = args.x_data_key
    y_data_key = args.y_data_key

    # Parse name_order if provided
    name_order = None
    if args.name_order:
        name_order = [name.strip() for name in args.name_order.split(",")]
        print(f"Using specified order: {name_order}")
    else:
        print("Using all available method names")

    # Handle prefix_filter (argparse provides a list if nargs is used)
    prefix_filter = args.prefix_filter
    if prefix_filter:
        print(f"Using prefix filter(s): {prefix_filter}")

    plot_afhp(
        name_order,
        eval_dir,
        prefix_filter,
        x_data_key,
        y_data_key,
        args.disable_horizontal_lines,
        args.key_filter,
        args.ablation_key,
        args.separate_figures,
        args.min_timestamp,
    )


if __name__ == "__main__":
    eval_result_plotter()

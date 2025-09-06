#!/usr/bin/env python3
"""
Utility functions for analyzing environments and agents.
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore

import argparse


# Add YRC to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from lib.procgenAISC.procgen import ProcgenEnv
from YRC.envs.procgen.wrappers import (
    VecExtractDictObs,
    TransposeFrame,
    ScaledFloatFrame,
    HardResetWrapper,
)


def create_env(random_percent: int = 100, start_level: int = 0, num_levels: int = 1):
    """
    Create a coinrun environment with specified parameters.

    Args:
        random_percent: Percentage of coin randomization (0=deterministic, 100=fully
            random)
        start_level: Starting level seed
        num_levels: Number of levels to include

    Returns:
        Wrapped procgen environment
    """
    # Create base environment
    env = ProcgenEnv(
        env_name="coinrun",
        num_envs=1,
        num_threads=1,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        rand_seed=start_level,  # Use level as seed for consistency
        use_backgrounds=True,
        use_monochrome_assets=False,
        restrict_themes=False,
        random_percent=random_percent,  # Key parameter for counterfactual analysis
    )

    # Apply wrappers (same as in YRC framework)
    env = VecExtractDictObs(env, "rgb")
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    env = HardResetWrapper(env)
    env.obs_shape = env.observation_space.shape

    return env


def plot_afhp(
    name_order: Optional[List[str]],
    extract_x_and_y_values_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    eval_file_dir: Path,
    prefix_filter: Optional[str],
    x_label: str,
    x_data_key: str,
    y_data_key: str,
):
    """
    Plot AFHP (Ask for Help Percentage) vs performance.

    Args:
        name_order: List of method names to plot in order. If None, uses all available
        names.
    """
    results = extract_results(eval_file_dir, prefix_filter)

    # Collect first and last performance values for all curves
    first_performances = []
    last_performances = []

    name_map = {
        "latent-svdd": "Latent SVDD",
        "random": "Random",
        "patient-ae": "Autoencoder",
        "center-focused": "Center-focused AE",
        "deep-svdd": "DeepSVDD",
    }

    # If name_order is None, use all available names
    if name_order is None:
        name_order = list(results.keys())

    # Clear previous plot
    plt.clf()

    for name in name_order:
        if name not in results:
            print(f"Warning: {name} not found in evals, skipping...")
            continue

        data_path = results[name]

        eval_data = np.load(data_path, allow_pickle=True)
        x, y = extract_x_and_y_values_fn(eval_data, x_data_key, y_data_key)

        # desired_percentiles = eval_data["desired_percentiles"]

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
    plt.title("Mean return vs. ask for help percentage")
    plt.legend()
    # plt.savefig("afhp_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def eval_result_plotter(
    extract_x_and_y_values_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    x_label: str,
):
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
        "--eval_file_dir",
        type=str,
        help="Directory containing the evaluation files.",
    )

    parser.add_argument(
        "--prefix_filter",
        default=None,
        type=str,
        help="Prefix filter for the evaluation files.",
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

    args = parser.parse_args()

    eval_file_dir = Path(args.eval_file_dir)
    prefix_filter = args.prefix_filter
    x_data_key = args.x_data_key
    y_data_key = args.y_data_key

    # Parse name_order if provided
    name_order = None
    if args.name_order:
        name_order = [name.strip() for name in args.name_order.split(",")]
        print(f"Using specified order: {name_order}")
    else:
        print("Using all available method names")

    plot_afhp(
        name_order,
        extract_x_and_y_values_fn,
        eval_file_dir,
        prefix_filter,
        x_label,
        x_data_key,
        y_data_key,
    )


def extract_results(
    eval_file_dir: Path, prefix_filter: Optional[str]
) -> dict[str, Path]:
    evals = {}

    for child in eval_file_dir.iterdir():
        if child.is_dir():
            method_name = child.name
            if (child / "eval_runs").exists():
                for grandchild in (child / "eval_runs").iterdir():
                    for grandgrandchild in grandchild.iterdir():
                        if (
                            grandgrandchild.is_file()
                            and grandgrandchild.suffix == ".npz"
                        ):
                            if prefix_filter is None or grandchild.stem.startswith(
                                f"{prefix_filter}"
                            ):
                                evals[method_name] = grandgrandchild
    return evals

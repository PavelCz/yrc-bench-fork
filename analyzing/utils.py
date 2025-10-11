#!/usr/bin/env python3
"""
Utility functions for analyzing environments and agents.
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore

import argparse
import json


# Add YRC to path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from lib.procgenAISC.procgen import ProcgenEnv
from YRC.envs.procgen.wrappers import (
    VecExtractDictObs,
    TransposeFrame,
    ScaledFloatFrame,
    HardResetWrapper,
)
import matplotlib

# The backend must be set before any plotting operations, but after importing
# procgen, which pulls in dependencies that set the backend.
matplotlib.use("TkAgg")


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
    eval_dir: Path,
    prefix_filter: Optional[str],
    x_data_key: str,
    y_data_key: str,
    disable_horizontal_lines: bool = False,
    key_filter: Optional[List[str]] = None,
    ablation_key: Optional[str] = None,
):
    """
    Plot AFHP (Ask for Help Percentage) vs performance.

    Args:
        name_order: List of method names to plot in order. If None, uses all available
        names.
        ablation_key: Config key to differentiate multiple runs from the same method.
    """
    results = extract_results(eval_dir, prefix_filter, ablation_key)

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

    # Clear previous plot
    plt.clf()

    for name in name_order:
        if name not in results:
            print(f"Warning: {name} not found in evals, skipping...")
            continue

        data_path = results[name]

        eval_data = np.load(data_path, allow_pickle=True)
        x, y = extract_x_and_y_values(eval_data, x_data_key, y_data_key)

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
        help=(
            "Config key (with dots for nested keys) to differentiate multiple runs "
            "from the same method. E.g., 'evaluation.coverage_fraction' or 'general.seed'"
        ),
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
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
        eval_dir,
        prefix_filter,
        x_data_key,
        y_data_key,
        args.disable_horizontal_lines,
        args.key_filter,
        args.ablation_key,
    )


def extract_from_data(data, key: str) -> np.ndarray:
    # Canonicalize these values by turning them into integers.
    level_ood_gt = [
        element["summary"]["test"]["level_ood_gt"] for element in data["meta"]
    ]

    level_ood_pred = [
        element["summary"]["test"]["level_ood_pred"] for element in data["meta"]
    ]
    if key == "ood_pred_percentage":
        pred_percentages = []
        for preds in level_ood_pred:
            percentage = sum(preds) / len(preds)
            pred_percentages.append(percentage)
        return np.array(pred_percentages)
    elif key == "ood_accuracy":
        accs = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            acc = (np.array(preds) == np.array(gts)).mean()
            accs.append(acc)
        return np.array(accs)
    elif key == "true_positive":
        tps = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            tp_count = ((np.array(preds) == 1)[np.array(gts) == 1]).sum()
            pos_count = (np.array(gts) == 1).sum()
            tps.append(tp_count / pos_count)
        return np.array(tps)
    elif key == "false_positive":
        fps = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            fp_count = ((np.array(preds) == 1)[np.array(gts) == 0]).sum()
            neg_count = (np.array(gts) == 0).sum()
            fps.append(fp_count / neg_count)
        return np.array(fps)
    elif key == "true_negative":
        tns = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            tn_count = ((np.array(preds) == 0)[np.array(gts) == 0]).sum()
            neg_count = (np.array(gts) == 0).sum()
            tns.append(tn_count / neg_count)
        return np.array(tns)
    elif key == "false_negative":
        fns = []
        for preds, gts in zip(level_ood_pred, level_ood_gt):
            fn_count = ((np.array(preds) == 0)[np.array(gts) == 1]).sum()
            pos_count = (np.array(gts) == 1).sum()
            fns.append(fn_count / pos_count)
        return np.array(fns)
    elif key == "performance":
        return data["performances"]
    elif key == "afhp":
        return data["afhps"]
    else:
        raise ValueError(f"Invalid key: {key}")


def extract_x_and_y_values(
    data, x_data_key: str, y_data_key: str
) -> tuple[np.ndarray, np.ndarray]:
    # x = data["afhps"]
    x = extract_from_data(data, x_data_key)
    y = extract_from_data(data, y_data_key)
    return x, y


def get_nested_config_value(config: dict, key_path: str):
    """
    Get a value from a nested config dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., "general.seed")
    
    Returns:
        The value at the key path, or None if not found
    """
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def extract_results(
    eval_dir: Path, prefix_filter: Optional[str], ablation_key: Optional[str] = None
) -> dict[str, Path]:
    evals = {}

    for child in eval_dir.iterdir():
        # Every child of the eval_dir is a different "grouped run".
        # We want to only select the runs that support the prefix_filter.
        if child.is_dir() and prefix_filter is not None and prefix_filter in child.name:
            for grandchild in child.iterdir():
                # Every grandchild is a different method.
                method_name = grandchild.name
                for run_dir in grandchild.iterdir():
                    # The runs_dirs are different runs with different 
                    # timestamps. Potentially, they might have different 
                    # hyperparameters.
                    for run_file in run_dir.iterdir():
                        if run_file.is_file() and run_file.suffix == ".npz":
                            # If ablation_key is provided, differentiate runs by config value
                            if ablation_key is not None:
                                config_file = run_dir / "config.json"
                                if config_file.exists():
                                    with open(config_file, 'r') as f:
                                        config = json.load(f)
                                    ablation_value = get_nested_config_value(config, ablation_key)
                                    # Create a unique key using method name and ablation value
                                    # Use only the final element of the key path for cleaner labels
                                    key_label = ablation_key.split('.')[-1]
                                    unique_key = f"{method_name}_{key_label}={ablation_value}"
                                    evals[unique_key] = run_file
                                else:
                                    # Fallback if config.json doesn't exist
                                    evals[method_name] = run_file
                            else:
                                evals[method_name] = run_file
    return evals

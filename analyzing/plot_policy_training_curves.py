from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLS = ["timesteps", "mean_episode_rewards", "val_mean_episode_rewards"]

# Each panel is (ylabel, list of (column, color, label, marker)).
RETURN_PANEL = (
    "Mean episode return",
    [
        ("mean_episode_rewards", "tab:blue", "Training", "o"),
        ("val_mean_episode_rewards", "tab:red", "Validation (pooled)", "s"),
    ],
)
ID_OOD_RETURN_PANEL = (
    "Val mean episode return",
    [
        ("val_id_mean_episode_rewards", "tab:green", "Validation ID", "o"),
        ("val_ood_mean_episode_rewards", "tab:orange", "Validation OOD", "s"),
    ],
)
ORACLE_REGRET_PANEL = (
    "Oracle regret",
    [
        ("val_mean_oracle_regret", "black", "Overall", "^"),
        ("val_id_mean_oracle_regret", "tab:green", "ID", "o"),
        ("val_ood_mean_oracle_regret", "tab:orange", "OOD", "s"),
    ],
)
PANELS = (RETURN_PANEL, ID_OOD_RETURN_PANEL, ORACLE_REGRET_PANEL)


def _smooth_series(values: pd.Series, window: int) -> pd.Series:
    """Return a centered rolling mean, or the original series if window <= 1."""
    if window <= 1 or len(values) < window:
        return values
    return values.rolling(window=window, min_periods=1, center=True).mean()


def load_training_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_path}: missing columns {missing_cols}; available: {list(df.columns)}"
        )
    return df


def aggregate_metric(dfs: Sequence[pd.DataFrame], column: str) -> pd.DataFrame:
    """Align runs on timesteps and return mean / IQR for one reward column."""
    series_list = []
    for i, df in enumerate(dfs):
        if column not in df.columns:
            continue
        mask = df[column].notna()
        if not mask.any():
            continue
        series = df.loc[mask].set_index("timesteps")[column].astype(float)
        series.name = f"run_{i}"
        series_list.append(series)

    if not series_list:
        return pd.DataFrame(columns=["mean", "q25", "q75"])

    aligned = pd.concat(series_list, axis=1)
    return pd.DataFrame(
        {
            "mean": aligned.mean(axis=1),
            "q25": aligned.quantile(0.25, axis=1),
            "q75": aligned.quantile(0.75, axis=1),
        }
    ).sort_index()


def _plot_aggregated_curve(
    ax,
    stats: pd.DataFrame,
    *,
    color: str,
    label: str,
    marker: str,
    smooth_window: int,
    show_iqr: bool,
):
    if stats.empty:
        return

    timesteps = stats.index
    mean = _smooth_series(stats["mean"], smooth_window)
    if show_iqr:
        q25 = _smooth_series(stats["q25"], smooth_window)
        q75 = _smooth_series(stats["q75"], smooth_window)
        ax.fill_between(timesteps, q25, q75, color=color, alpha=0.2, linewidth=0)
        ax.plot(timesteps, mean, label=label, color=color, linewidth=2.5)
        return

    rewards = mean
    if smooth_window > 1:
        ax.plot(
            timesteps,
            stats["mean"],
            color=color,
            linewidth=1,
            alpha=0.25,
            marker=marker,
            markersize=3,
        )
        ax.plot(timesteps, rewards, label=label, color=color, linewidth=2.5)
    else:
        ax.plot(
            timesteps,
            rewards,
            label=label,
            linewidth=2,
            marker=marker,
            markersize=4,
            alpha=0.8,
            color=color,
        )


def _curve_label(base: str, n_runs: int, show_iqr: bool) -> str:
    if show_iqr:
        return f"{base} (n={n_runs})"
    return base


def _active_panels(
    dfs: Sequence[pd.DataFrame],
) -> List[Tuple[str, Sequence[Tuple[str, str, str, str]]]]:
    active = []
    for ylabel, curves in PANELS:
        if any(not aggregate_metric(dfs, column).empty for column, _, _, _ in curves):
            active.append((ylabel, curves))
    return active


def plot_training_curves(
    csv_paths: Sequence[Union[str, Path]],
    output_path="training_curves.png",
    show_plot=True,
    smooth_window: int = 0,
):
    """Plot training/validation returns and optional ID/OOD and regret panels."""
    dfs: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        try:
            df = load_training_csv(path)
        except Exception as e:
            print(f"Error reading CSV file {path}: {e}")
            continue
        print(f"Loaded {path} with {len(df)} rows")
        dfs.append(df)

    if not dfs:
        print("No valid CSV files loaded")
        return None, None

    panels = _active_panels(dfs)
    if not panels:
        print("No valid data points found for either curve")
        return None, None

    n_runs = len(dfs)
    show_iqr = n_runs > 1
    n_panels = len(panels)
    plt.style.use("default")
    fig, axes = plt.subplots(
        n_panels,
        1,
        sharex=True,
        figsize=(12, 4.2 * n_panels),
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, (ylabel, curves) in zip(axes, panels):
        for column, color, label, marker in curves:
            stats = aggregate_metric(dfs, column)
            if stats.empty:
                continue
            _plot_aggregated_curve(
                ax,
                stats,
                color=color,
                label=_curve_label(label, n_runs, show_iqr),
                marker=marker,
                smooth_window=smooth_window,
                show_iqr=show_iqr,
            )
            print(f"{column}: {len(stats)} points")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=11)

    axes[-1].set_xlabel("Timesteps", fontsize=14)
    title = "Policy training curves"
    if show_iqr:
        title = f"Policy training curves (mean + IQR, n={n_runs})"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved as: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    if show_plot:
        plt.show()

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from CSV log files"
    )
    parser.add_argument(
        "--csv_path",
        nargs="+",
        required=True,
        help="Path(s) to CSV log files. Multiple files are averaged at each timestep.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_curves.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the plot interactively"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Rolling-average window in log points (default: 0, no smoothing)",
    )

    args = parser.parse_args()

    missing = [path for path in args.csv_path if not Path(path).exists()]
    if missing:
        print("CSV file(s) not found:")
        for path in missing:
            print(f"  {path}")
        return

    fig, ax = plot_training_curves(
        args.csv_path,
        args.output,
        not args.no_show,
        smooth_window=args.smooth,
    )

    if fig is None:
        print("Failed to create plot")
        return


if __name__ == "__main__":
    main()

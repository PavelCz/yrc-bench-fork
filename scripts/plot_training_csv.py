#!/usr/bin/env python3
import argparse
from pathlib import Path


DEFAULT_METRICS = [
    "mean_episode_rewards",
    "val_mean_episode_rewards",
    "val_random_start_mean_episode_rewards",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Procgen policy training CSVs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, CSV files, or experiment directories containing runs.",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="CSV columns to plot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to plot.png for one run, otherwise training_curves.png.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling mean window in logged rows.",
    )
    return parser.parse_args()


def find_csvs(paths):
    csvs = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path not in seen:
                csvs.append(path)
                seen.add(path)
            continue

        direct_csv = path / "log-append.csv"
        if direct_csv.exists():
            if direct_csv not in seen:
                csvs.append(direct_csv)
                seen.add(direct_csv)
            continue

        for csv_path in sorted(path.rglob("log-append.csv")):
            if csv_path not in seen:
                csvs.append(csv_path)
                seen.add(csv_path)

    if not csvs:
        raise FileNotFoundError("No log-append.csv files found.")
    return csvs


def label_for(csv_path):
    run_dir = csv_path.parent
    exp_dir = run_dir.parent
    return f"{exp_dir.name}/{run_dir.name}"


def read_run(csv_path, smooth):
    import pandas as pd

    data = pd.read_csv(csv_path)
    if "timesteps" not in data.columns:
        raise ValueError(f"{csv_path} does not contain a timesteps column.")

    if smooth > 1:
        numeric_cols = data.select_dtypes("number").columns
        data[numeric_cols] = data[numeric_cols].rolling(smooth, min_periods=1).mean()
    return data


def plot():
    args = parse_args()
    import matplotlib.pyplot as plt

    csvs = find_csvs(args.paths)
    runs = [(csv_path, read_run(csv_path, args.smooth)) for csv_path in csvs]
    metrics = [
        metric
        for metric in args.metrics
        if any(metric in run_data.columns for _, run_data in runs)
    ]
    if not metrics:
        available = sorted(set().union(*(run_data.columns for _, run_data in runs)))
        raise ValueError(
            "None of the requested metrics were found. "
            f"Available columns: {', '.join(available)}"
        )

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(12, max(4, 3.2 * len(metrics))),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, metric in zip(axes, metrics):
        for csv_path, run_data in runs:
            if metric not in run_data.columns:
                continue
            ax.plot(run_data["timesteps"], run_data[metric], label=label_for(csv_path))
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("timesteps")
    axes[0].legend(fontsize="small", loc="best")
    fig.tight_layout()

    output = args.output
    if output is None:
        output = (
            csvs[0].parent / "plot.png"
            if len(csvs) == 1
            else Path("training_curves.png")
        )
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    plot()

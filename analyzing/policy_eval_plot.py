#!/usr/bin/env python3
"""
Aggregate and plot eval_policy.py results across experiment IDs.

Expected directory structure:
    <eval_dir>/<prefix>_<env>_<agent>_exp0/<config_name>/<timestamp>/policy_eval_results.json
    <eval_dir>/<prefix>_<env>_<agent>_exp1/<config_name>/<timestamp>/policy_eval_results.json
    ...

This script:
- finds the latest policy_eval_results.json for each matching experiment directory
- aggregates overall / ID / OOD returns across experiment IDs
- prints a console summary
- optionally saves a grouped bar plot
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
METRIC_ORDER = ["overall", "id", "ood"]
METRIC_LABELS = {
    "overall": "Overall",
    "id": "ID",
    "ood": "OOD",
}
SERIES_COLORS = {
    "strong": "#1f77b4",
    "weak": "#ff7f0e",
    "sim": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot eval_policy.py results across experiment IDs"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing eval_policy results (typically experiments/evals)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        nargs="+",
        default=None,
        help="Prefix filter(s) for experiment directories",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["coinrun", "maze", "maze_afh", "heist"],
        help="Environment filter",
    )
    parser.add_argument(
        "--agent_filter",
        "-f",
        type=str,
        nargs="+",
        default=None,
        choices=["sim", "weak", "strong"],
        help="Agents to include",
    )
    parser.add_argument(
        "--agent_order",
        "-a",
        type=str,
        default=None,
        help="Comma-separated list giving display order for agents/series",
    )
    parser.add_argument(
        "--use_pooled",
        action="store_true",
        help="Plot pooled episode means instead of run-level means with SEM error bars",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the figure (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save aggregated summary as JSON",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available series and exit",
    )
    return parser.parse_args()


def configure_matplotlib(show_plot: bool):
    import matplotlib

    if show_plot:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    try:
        from analyzing.plotting_common import setup_plot_style, style_plot_for_publication

        setup_plot_style(paper_mode=False, use_latex=False)
        return plt, style_plot_for_publication
    except Exception:
        return plt, None


def parse_experiment_dir(dir_name: str) -> Optional[Tuple[str, str, str, int]]:
    """Parse <prefix>_<env>_<agent>_expN directory names."""
    pattern = r"^(.+)_(coinrun|maze|maze_afh|heist)_(sim|weak|strong)_exp(\d+)$"
    match = re.match(pattern, dir_name)
    if match:
        prefix = match.group(1)
        env = match.group(2)
        agent = match.group(3)
        exp_id = int(match.group(4))
        return prefix, env, agent, exp_id
    return None


def _result_sort_key(path: Path) -> Tuple[int, str]:
    run_dir = path.parent
    if TIMESTAMP_PATTERN.match(run_dir.name):
        return (1, run_dir.name)
    return (0, f"{run_dir.stat().st_mtime:020.6f}")


def find_latest_policy_eval_result(experiment_dir: Path) -> Optional[Path]:
    """Find the newest policy_eval_results.json under an experiment group directory."""
    candidates = [
        path
        for path in experiment_dir.rglob("policy_eval_results.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_result_sort_key)[-1]


def extract_policy_eval_results(
    eval_dir: Path,
    prefix_filter: Optional[List[str]] = None,
    env_filter: Optional[str] = None,
    agent_filter: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[int, Path]], Dict[str, Dict[str, str]]]:
    """
    Extract the latest result file for each matching experiment id.

    Returns:
        results: mapping from series label to {exp_id: result_path}
        metadata: mapping from series label to metadata like prefix/env/agent
    """
    include_prefix_in_label = prefix_filter is None or len(prefix_filter) != 1
    results: Dict[str, Dict[int, Path]] = defaultdict(dict)
    metadata: Dict[str, Dict[str, str]] = {}

    for child in eval_dir.iterdir():
        if not child.is_dir():
            continue

        parsed = parse_experiment_dir(child.name)
        if parsed is None:
            continue

        prefix, env, agent, exp_id = parsed

        if prefix_filter is not None and prefix not in prefix_filter:
            continue
        if env_filter is not None and env != env_filter:
            continue
        if agent_filter is not None and agent not in agent_filter:
            continue

        result_path = find_latest_policy_eval_result(child)
        if result_path is None:
            continue

        label = f"{prefix}:{agent}" if include_prefix_in_label else agent
        results[label][exp_id] = result_path
        metadata[label] = {"prefix": prefix, "env": env, "agent": agent}

    return dict(results), metadata


def compute_return_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute count/mean/std/sem for a list of floats."""
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "sem": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    sem = float(std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": sem,
    }


def load_policy_eval_result(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def aggregate_series(result_files: Dict[int, Path]) -> Dict:
    """Aggregate multiple policy_eval_results.json files for one series."""
    per_run = []
    run_metric_values: Dict[str, List[float]] = {metric: [] for metric in METRIC_ORDER}
    pooled_returns: List[float] = []
    pooled_id_returns: List[float] = []
    pooled_ood_returns: List[float] = []

    for exp_id in sorted(result_files):
        path = result_files[exp_id]
        data = load_policy_eval_result(path)

        all_returns = [float(x) for x in data.get("all_returns", [])]
        level_ood_gt = data.get("level_ood_gt")

        if level_ood_gt is not None and len(level_ood_gt) != len(all_returns):
            raise ValueError(
                f"Mismatched lengths in {path}: "
                f"{len(all_returns)} returns vs {len(level_ood_gt)} OOD labels"
            )

        if level_ood_gt is None:
            id_returns: List[float] = []
            ood_returns: List[float] = []
        else:
            id_returns = [
                ret for ret, is_ood in zip(all_returns, level_ood_gt) if not bool(is_ood)
            ]
            ood_returns = [
                ret for ret, is_ood in zip(all_returns, level_ood_gt) if bool(is_ood)
            ]

        overall_stats = compute_return_stats(all_returns)
        id_stats = compute_return_stats(id_returns)
        ood_stats = compute_return_stats(ood_returns)

        per_run.append(
            {
                "exp_id": exp_id,
                "path": str(path),
                "num_episodes": overall_stats["count"],
                "mean_return": overall_stats["mean"],
                "id_num_episodes": id_stats["count"],
                "id_mean_return": id_stats["mean"],
                "ood_num_episodes": ood_stats["count"],
                "ood_mean_return": ood_stats["mean"],
            }
        )

        if overall_stats["mean"] is not None:
            run_metric_values["overall"].append(float(overall_stats["mean"]))
        if id_stats["mean"] is not None:
            run_metric_values["id"].append(float(id_stats["mean"]))
        if ood_stats["mean"] is not None:
            run_metric_values["ood"].append(float(ood_stats["mean"]))

        pooled_returns.extend(all_returns)
        pooled_id_returns.extend(id_returns)
        pooled_ood_returns.extend(ood_returns)

    run_stats = {
        metric: compute_return_stats(values)
        for metric, values in run_metric_values.items()
    }
    pooled_stats = {
        "overall": compute_return_stats(pooled_returns),
        "id": compute_return_stats(pooled_id_returns),
        "ood": compute_return_stats(pooled_ood_returns),
    }

    return {
        "num_runs": len(per_run),
        "per_run": per_run,
        "run_stats": run_stats,
        "pooled_stats": pooled_stats,
    }


def list_available_series(eval_dir: Path, prefix: Optional[List[str]], env: Optional[str]) -> None:
    results, metadata = extract_policy_eval_results(eval_dir, prefix, env, None)
    if not results:
        print("No policy-eval series found.")
        return

    print("Available series:")
    for label in sorted(results):
        meta = metadata[label]
        exp_ids = sorted(results[label].keys())
        print(
            f"  {label}: prefix={meta['prefix']} env={meta['env']} "
            f"agent={meta['agent']} exp_ids={exp_ids}"
        )


def print_summary(series_results: Dict[str, Dict], metadata: Dict[str, Dict[str, str]]) -> None:
    """Print aggregated stats to the console."""
    for label, summary in series_results.items():
        meta = metadata[label]
        print(f"\n{label}")
        print(
            f"  prefix={meta['prefix']} env={meta['env']} agent={meta['agent']} "
            f"runs={summary['num_runs']}"
        )

        for metric in METRIC_ORDER:
            run_stats = summary["run_stats"][metric]
            pooled_stats = summary["pooled_stats"][metric]

            run_mean = (
                f"{run_stats['mean']:.4f} +/- {run_stats['sem']:.4f}"
                if run_stats["mean"] is not None and run_stats["sem"] is not None
                else "n/a"
            )
            pooled_mean = (
                f"{pooled_stats['mean']:.4f} (n={pooled_stats['count']})"
                if pooled_stats["mean"] is not None
                else "n/a"
            )
            print(
                f"  {metric:>7}: run-mean={run_mean}  pooled={pooled_mean}"
            )

        print("  per-run:")
        for run in summary["per_run"]:
            overall = (
                f"{run['mean_return']:.4f}" if run["mean_return"] is not None else "n/a"
            )
            id_mean = (
                f"{run['id_mean_return']:.4f}" if run["id_mean_return"] is not None else "n/a"
            )
            ood_mean = (
                f"{run['ood_mean_return']:.4f}" if run["ood_mean_return"] is not None else "n/a"
            )
            print(
                f"    exp{run['exp_id']}: overall={overall} (n={run['num_episodes']}) "
                f"id={id_mean} (n={run['id_num_episodes']}) "
                f"ood={ood_mean} (n={run['ood_num_episodes']})"
            )


def build_series_order(
    available_labels: List[str],
    metadata: Dict[str, Dict[str, str]],
    agent_order: Optional[str],
) -> List[str]:
    if agent_order is None:
        return sorted(available_labels)

    requested = [item.strip() for item in agent_order.split(",") if item.strip()]
    requested_labels = set(requested)

    ordered = []
    for item in requested:
        if item in available_labels:
            ordered.append(item)
            continue

        # Allow bare agent names when there is a single matching label.
        matches = [
            label for label in available_labels if metadata[label]["agent"] == item
        ]
        if len(matches) == 1:
            ordered.append(matches[0])
            requested_labels.add(matches[0])

    for label in available_labels:
        if label not in ordered and label not in requested_labels:
            ordered.append(label)

    return ordered


def plot_summary(
    series_results: Dict[str, Dict],
    metadata: Dict[str, Dict[str, str]],
    ordered_labels: List[str],
    title: Optional[str],
    save_path: Optional[Path],
    show_plot: bool,
    use_pooled: bool,
) -> None:
    plt, style_plot_for_publication = configure_matplotlib(show_plot)

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(METRIC_ORDER))
    width = 0.8 / max(len(ordered_labels), 1)

    for idx, label in enumerate(ordered_labels):
        summary = series_results[label]
        meta = metadata[label]
        label_name = label

        if use_pooled:
            heights = [
                summary["pooled_stats"][metric]["mean"] or 0.0 for metric in METRIC_ORDER
            ]
            yerr = None
        else:
            heights = [
                summary["run_stats"][metric]["mean"] or 0.0 for metric in METRIC_ORDER
            ]
            yerr = [
                summary["run_stats"][metric]["sem"] or 0.0 for metric in METRIC_ORDER
            ]

        offset = (idx - (len(ordered_labels) - 1) / 2) * width
        color = SERIES_COLORS.get(meta["agent"])
        ax.bar(
            x + offset,
            heights,
            width=width,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            label=label_name,
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in METRIC_ORDER])
    ax.set_ylabel("Average Return")

    if title is not None:
        ax.set_title(title)
    else:
        prefixes = sorted({metadata[label]["prefix"] for label in ordered_labels})
        envs = sorted({metadata[label]["env"] for label in ordered_labels})
        mode = "Pooled episode means" if use_pooled else "Run mean +/- SEM"
        ax.set_title(f"Policy Eval Summary ({', '.join(prefixes)} / {', '.join(envs)})\n{mode}")

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if style_plot_for_publication is not None:
        style_plot_for_publication(
            ax=ax,
            legend_outside=False,
        )
    else:
        ax.legend(frameon=False)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to {save_path}")
    elif show_plot:
        plt.show()

    plt.close(fig)


def save_summary_json(
    series_results: Dict[str, Dict],
    metadata: Dict[str, Dict[str, str]],
    output_path: Path,
) -> None:
    output = {"series": {}}
    for label, summary in series_results.items():
        output["series"][label] = {
            "metadata": metadata[label],
            **summary,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved summary JSON to {output_path}")


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    if args.list:
        list_available_series(eval_dir, args.prefix, args.env)
        return 0

    results, metadata = extract_policy_eval_results(
        eval_dir,
        prefix_filter=args.prefix,
        env_filter=args.env,
        agent_filter=args.agent_filter,
    )

    if not results:
        print("No matching policy evaluation results found.")
        return 1

    ordered_labels = build_series_order(
        sorted(results.keys()), metadata, args.agent_order
    )

    series_results = {
        label: aggregate_series(results[label]) for label in ordered_labels
    }

    print_summary(series_results, metadata)

    if args.save_json is not None:
        save_summary_json(series_results, metadata, Path(args.save_json))

    if args.save is not None or not args.list:
        plot_summary(
            series_results,
            metadata,
            ordered_labels,
            title=args.title,
            save_path=Path(args.save) if args.save is not None else None,
            show_plot=args.save is None,
            use_pooled=args.use_pooled,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the paper's acting-policy table and labeled bar plot.

Run with ``python -m analyzing.plot_policy_result_summary --paper-dir PAPER_REPO``.
The table is included by the appendix; the plot is saved as a PDF in ``img``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from analyzing.paper_plot import extract_icml_results
from analyzing.plot_main_figures import PANELS


DEFAULT_EVAL_DIR = Path("/home/pavel/data/goal-misgen/icml-evals")
LEVEL_TYPES = ("id", "ood", "all")
LEVEL_LABELS = {"id": "ID", "ood": "OOD", "all": "Mean"}
AGENTS = ("novice", "expert")


@dataclass(frozen=True)
class Panel:
    name: str
    latex_name: str
    env: str
    prefixes: tuple[str, ...]
    method: str
    seeds: tuple[int, ...]
    ymax: float


def _main_prefixes(env: str) -> tuple[str, ...]:
    return tuple(next(panel["prefix"] for panel in PANELS if panel["env"] == env))


PANELS_TO_SUMMARIZE = (
    Panel(
        "Maze (robust)",
        r"\maze\ (robust)",
        "maze",
        _main_prefixes("maze"),
        "max-prob_robust400",
        tuple(range(8)),
        10.8,
    ),
    Panel(
        "Coinrun",
        r"\coin",
        "coinrun",
        _main_prefixes("coinrun"),
        "max-prob",
        tuple(range(8)),
        10.8,
    ),
    Panel(
        "K&C",
        r"\kandc",
        "heist",
        _main_prefixes("heist"),
        "max-prob",
        tuple(range(8)),
        4.1,
    ),
    Panel(
        "Maze (traditional)",
        r"\maze\ (traditional)",
        "maze",
        ("tmlr-maze-reval",),
        "max-prob",
        tuple(range(4)),
        10.8,
    ),
)


@dataclass(frozen=True)
class ReturnStats:
    mean: float
    sd: float
    per_seed: tuple[float, ...]


def summarize_panel(eval_dir: Path, panel: Panel) -> dict[str, dict[str, ReturnStats]]:
    """Use one endpoint evaluation per trained policy seed, never pool seeds."""
    paths = extract_icml_results(eval_dir, list(panel.prefixes), panel.env)
    method_runs = paths.get(panel.method, {})
    per_seed: dict[str, dict[str, list[float]]] = {
        agent: {level_type: [] for level_type in LEVEL_TYPES} for agent in AGENTS
    }

    for seed in panel.seeds:
        if seed not in method_runs:
            raise ValueError(f"Missing {panel.name} {panel.method} experiment {seed}")
        path = method_runs[seed].parent / "eval_seed_1_test.npz"
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=True) as data:
            afhps = np.asarray(data["afhps"], dtype=float)
            if afhps.size == 0:
                raise ValueError(f"No AFHP endpoints in {path}")
            for agent, index, target in (
                ("novice", int(np.argmin(afhps)), 0.0),
                ("expert", int(np.argmax(afhps)), 1.0),
            ):
                if not np.isclose(afhps[index], target):
                    raise ValueError(f"Missing {agent} endpoint in {path}")
                summary = data["meta"][index]["summary"]["test"]
                returns = np.asarray(summary["raw_returns"], dtype=float)
                ood = np.asarray(summary["level_ood_gt"], dtype=bool)
                if returns.size == 0 or returns.size != ood.size:
                    raise ValueError(f"Invalid return/level labels in {path}")
                for level_type, mask in (
                    ("id", ~ood),
                    ("ood", ood),
                    ("all", np.ones_like(ood, dtype=bool)),
                ):
                    if not mask.any():
                        raise ValueError(f"No {level_type} episodes in {path}")
                    per_seed[agent][level_type].append(float(returns[mask].mean()))

    return {
        agent: {
            level_type: ReturnStats(
                mean=float(np.mean(per_seed[agent][level_type])),
                sd=float(np.std(per_seed[agent][level_type], ddof=1)),
                per_seed=tuple(per_seed[agent][level_type]),
            )
            for level_type in LEVEL_TYPES
        }
        for agent in AGENTS
    }


def summarize_all(eval_dir: Path) -> dict[str, dict[str, dict[str, ReturnStats]]]:
    return {
        panel.name: summarize_panel(eval_dir, panel) for panel in PANELS_TO_SUMMARIZE
    }


def write_latex_table(
    summaries: dict[str, dict[str, dict[str, ReturnStats]]], path: Path
) -> None:
    """Write a booktabs tabular for inclusion by the paper's existing table."""
    lines = [
        r"\begin{tabular}{ll|cc}",
        r"\toprule",
        r"Environment & Level Type & \textsc{Novice} & \textsc{Expert} \\",
        r"\midrule",
    ]
    for panel_index, panel in enumerate(PANELS_TO_SUMMARIZE):
        if panel_index == 3:
            lines.append(r"\midrule")
        elif panel_index:
            lines.append(r"\addlinespace")
        for row_index, level_type in enumerate(LEVEL_TYPES):
            environment = (
                rf"\multirow{{3}}{{*}}{{\textbf{{({chr(65 + panel_index)})}} {panel.latex_name}}}"
                if row_index == 0
                else ""
            )
            label = (
                rf"\textbf{{{LEVEL_LABELS[level_type]}}}"
                if level_type == "all"
                else LEVEL_LABELS[level_type]
            )
            cells = []
            for agent in AGENTS:
                stats = summaries[panel.name][agent][level_type]
                mean = f"{stats.mean:.2f}"
                if level_type == "all":
                    mean = rf"\mathbf{{{mean}}}"
                cells.append(rf"${mean}\pm {stats.sd:.2f}$")
            lines.append(f"{environment} & {label} & {cells[0]} & {cells[1]} " + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plot_bars(
    summaries: dict[str, dict[str, dict[str, ReturnStats]]],
    pdf_path: Path,
    png_path: Path | None,
) -> None:
    """Plot seed means with across-seed standard deviations and value labels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    colors = {"novice": "#c44e52", "expert": "#4c72b0"}
    x = np.arange(3)
    width = 0.36
    for ax, panel in zip(axes.flat, PANELS_TO_SUMMARIZE):
        for agent, offset in (("novice", -width / 2), ("expert", width / 2)):
            stats = [summaries[panel.name][agent][item] for item in LEVEL_TYPES]
            values = [item.mean for item in stats]
            errors = [item.sd for item in stats]
            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=colors[agent],
                label=agent.capitalize(),
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, value, error in zip(bars, values, errors):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + panel.ymax * 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title(f"{panel.name} (n={len(panel.seeds)} seeds)")
        ax.set_xticks(x, ("ID", "OOD", "Overall"))
        ax.set_ylim(0, panel.ymax)
        ax.set_ylabel("Episodic return")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--png-out", type=Path)
    args = parser.parse_args()

    summaries = summarize_all(args.eval_dir)
    table_path = args.paper_dir / "tables/policy-results.tex"
    plot_path = args.paper_dir / "img/policy-results-bars.pdf"
    write_latex_table(summaries, table_path)
    plot_bars(summaries, plot_path, args.png_out)
    print(f"Table: {table_path}")
    print(f"Bar plot: {plot_path}")
    for panel in PANELS_TO_SUMMARIZE:
        print(f"{panel.name}: {len(panel.seeds)} seeds")
        for agent in AGENTS:
            values = summaries[panel.name][agent]
            print(
                f"  {agent}: "
                + ", ".join(
                    f"{level_type}={values[level_type].mean:.2f}±{values[level_type].sd:.2f}"
                    for level_type in LEVEL_TYPES
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot median AUC across recoverable, penalty, and termination settings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyzing.paper_plot import (
    SHARED_LEGEND_METHODS,
    _RANDOM_LABEL,
    _method_line_handle,
    format_plot_label,
)
from analyzing.plotting_common import setup_plot_style

matplotlib.use("Agg", force=True)


DEFAULT_TABLE_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/tables"
)
DEFAULT_OUT_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/img"
)

SETTINGS: Tuple[str, ...] = ("recoverable", "penalty", "termination")
SETTING_LABELS: Tuple[str, ...] = ("Recoverable", "Penalty", "Termination")

SCNAME_TO_METHOD = {
    "PartialOracle": "oracle-lb-random",
    "Heuristic": "ts-random",
    "Ensemble": "ensemble-single",
    "MaxProb": "max-prob",
    "MaxLogit": "max-logit",
    "ImageSVDD": "svdd-image",
    "LatentSVDD": "svdd-latent",
}

MEDIAN_RE = re.compile(
    r"\\textsc\{(?P<name>[^}]+)\}(?:\})?\s*&\s*(?:\\textbf\{)?(?P<value>--|[0-9]+\.[0-9]+)"
)

PANELS = (
    {
        "title": r"\texttt{Coinrun}",
        "tables": {
            "recoverable": "auc-coinrun.tex",
            "penalty": "auc-coinrun-proxy-penalty.tex",
            "termination": "auc-coinrun-proxy-fail.tex",
        },
        "show_ylabel": True,
    },
    {
        "title": r"\texttt{Maze}",
        "tables": {
            "recoverable": "auc-maze.tex",
            "penalty": "auc-maze-proxy-penalty.tex",
            "termination": "auc-maze-proxy-fail.tex",
        },
        "show_ylabel": False,
    },
    {
        "title": r"\texttt{K\&C}",
        "tables": {
            "recoverable": "auc-kandc.tex",
            "penalty": "auc-kandc-proxy-penalty.tex",
            "termination": "auc-kandc-proxy-fail.tex",
        },
        "show_ylabel": False,
    },
)

SAVE_NAME = "auc-across-settings.pdf"
FIGSIZE = (24.0, 6.4)
AXIS_LABEL_SIZE = 36
TICK_LABEL_SIZE = 28
TITLE_SIZE = 32
RANDOM_LINESTYLE = (0, (2, 2, 10, 2))

LATEX_SNIPPET = r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{img/auc-across-settings.pdf}
    \caption{Median AUC as proxy pursuit becomes more costly, from the
    recoverable setting through a $-5$ penalty with continuation to
    termination. The dotted line is \textsc{Random} ($0.5$).}
    \label{fig:app:auc-across-settings}
\end{figure}
""".strip()


def parse_auc_table_medians(path: Path) -> Dict[str, float]:
    """Map canonical method names to median AUC values from an AUC tabular."""
    text = path.read_text()
    medians: Dict[str, float] = {}
    for match in MEDIAN_RE.finditer(text):
        scname = match.group("name")
        raw_value = match.group("value")
        method = SCNAME_TO_METHOD.get(scname)
        if method is None or raw_value == "--":
            continue
        medians[method] = float(raw_value)
    return medians


def load_panel_medians(
    table_dir: Path, panel: Mapping[str, object]
) -> Dict[str, Dict[str, float]]:
    tables = panel["tables"]
    if not isinstance(tables, dict):
        raise TypeError("panel['tables'] must be a dict of setting -> filename")
    return {
        setting: parse_auc_table_medians(table_dir / filename)
        for setting, filename in tables.items()
    }


def _method_ys(
    method: str, setting_medians: Mapping[str, Mapping[str, float]]
) -> List[float]:
    missing = [
        setting for setting in SETTINGS if method not in setting_medians[setting]
    ]
    if missing:
        raise KeyError(f"{method} missing AUC medians for settings: {missing}")
    return [setting_medians[setting][method] for setting in SETTINGS]


def plot_auc_across_settings(
    table_dir: Path,
    save_path: Path,
    panels: Sequence[Mapping[str, object]] = PANELS,
) -> None:
    setup_plot_style(paper_mode=True, use_latex=True)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=FIGSIZE,
        sharey=True,
    )
    n_methods = len(SHARED_LEGEND_METHODS)
    xs = list(range(len(SETTINGS)))
    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []

    for ax, panel in zip(axes, panels):
        setting_medians = load_panel_medians(table_dir, panel)
        for method_idx, method in enumerate(SHARED_LEGEND_METHODS):
            handle = _method_line_handle(method, method_idx, n_methods, paper_mode=True)
            ys = _method_ys(method, setting_medians)
            ax.plot(
                xs,
                ys,
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                linewidth=2,
                marker="o",
                markersize=9,
            )
            if ax is axes[0]:
                legend_handles.append(handle)
                legend_labels.append(format_plot_label(method, paper_mode=True))
        ax.axhline(
            0.5,
            color="black",
            linestyle=RANDOM_LINESTYLE,
            alpha=0.9,
            linewidth=2,
            zorder=0,
        )
        ax.set_xlim(-0.15, len(SETTINGS) - 1 + 0.15)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(list(SETTING_LABELS), rotation=18, ha="right")
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_title(str(panel["title"]), fontsize=TITLE_SIZE)
        if panel["show_ylabel"]:
            ax.set_ylabel("Median AUC", fontsize=AXIS_LABEL_SIZE)

    random_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle=RANDOM_LINESTYLE,
        alpha=0.9,
        linewidth=2,
    )
    legend_handles.append(random_handle)
    legend_labels.append(_RANDOM_LABEL)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=22,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.subplots_adjust(top=0.76, bottom=0.26, left=0.07, right=0.995, wspace=0.24)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write a three-panel median-AUC line plot across recoverable, "
            "penalty, and termination settings, reading existing AUC .tex tables."
        )
    )
    parser.add_argument(
        "--table-dir",
        "--table_dir",
        dest="table_dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"Directory of AUC tabular .tex files (default: {DEFAULT_TABLE_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the PDF (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    table_dir = args.table_dir
    if not table_dir.is_dir():
        parser.error(f"table_dir does not exist: {table_dir}")

    plot_auc_across_settings(table_dir, args.out_dir / SAVE_NAME)
    print("\nLaTeX:\n")
    print(LATEX_SNIPPET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

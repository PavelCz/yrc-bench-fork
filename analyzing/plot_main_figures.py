#!/usr/bin/env python3
"""Write the three main AFHP panels (no legends) plus a shared two-line legend."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from analyzing.paper_plot import plot_icml_results, save_shared_legend

matplotlib.use("Agg", force=True)


DEFAULT_EVAL_DIR = Path("/home/pavel/data/goal-misgen/icml-evals")

PANELS = (
    {
        "save_name": "coinrun-main-normalized.pdf",
        "prefix": [
            "tmlr2-svdd",
            "tmlr3-svdd",
            "tmlr4-svdd",
            "imcl04",
            "icml04",
            "tmlr-oracle-lb",
        ],
        "env": "coinrun",
        "robust_filter": "all",
        "calculate_auc": False,
    },
    {
        "save_name": "maze-main-normalized.pdf",
        "prefix": [
            "tmlr-robust-maze-1",
            "tmlr-robust-maze-3",
            "tmlr-oracle-lb-robust400",
            "tmlr-maze-svdd-fix",
        ],
        "env": "maze",
        "robust_filter": "robust",
        "calculate_auc": False,
    },
    {
        "save_name": "heist-main-normalized.pdf",
        "prefix": ["tmlr-heist02-metrics"],
        "env": "heist",
        "robust_filter": "all",
        "calculate_auc": True,
    },
)

SHARED_METHOD_FILTER = ["ensemble", "wait"]
LEGEND_NAME = "main-legend.pdf"

LATEX_SNIPPET = r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{img/main-legend.pdf}
    \vspace{0.4em}
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/coinrun-main-normalized.pdf}
        \caption{\texttt{Coinrun}.}
        \label{fig:results:coinrun}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/maze-main-normalized.pdf}
        \caption{\texttt{Maze}.}
        \label{fig:results:maze}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/heist-main-normalized.pdf}
        \caption{\texttt{Heist}.}
        \label{fig:results:heist}
    \end{subfigure}
    \caption{Return versus ask-for-help percentage (AFHP), normalized so the
    novice is 0 and the expert is 1. Shaded bands are the interquartile range
    across seeds. The legend applies to all three panels.}
    \label{fig:results}
\end{figure}
""".strip()


def _plot_panel(eval_dir: Path, out_dir: Path, panel: dict) -> None:
    save_path = out_dir / panel["save_name"]
    plot_icml_results(
        eval_dir=eval_dir,
        prefix_filter=panel["prefix"],
        env_filter=panel["env"],
        x_data_key="level_afhp",
        y_data_key="performance",
        method_filter=SHARED_METHOD_FILTER,
        save_path=str(save_path),
        paper_mode=True,
        calculate_auc=panel["calculate_auc"],
        robust_filter=panel["robust_filter"],
        normalize_y=True,
        show_legend=False,
    )
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write coinrun/maze/heist main AFHP panels without legends, "
            "plus a two-line shared legend PDF."
        )
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help=f"Directory containing evaluation files (default: {DEFAULT_EVAL_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=Path("img"),
        help="Directory for the four PDFs (default: img)",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.is_dir():
        parser.error(f"eval_dir does not exist: {eval_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for panel in PANELS:
        _plot_panel(eval_dir, out_dir, panel)

    save_shared_legend(str(out_dir / LEGEND_NAME), paper_mode=True)

    print("\nLaTeX:\n")
    print(LATEX_SNIPPET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

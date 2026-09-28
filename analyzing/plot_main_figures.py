#!/usr/bin/env python3
"""Write the three main AFHP panels, a shared legend, and AUC tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from analyzing.paper_plot import (
    AUC_ENV_COLUMN_HEADERS,
    plot_icml_results,
    save_shared_legend,
    write_auc_env_columns_table,
)

matplotlib.use("Agg", force=True)


DEFAULT_EVAL_DIR = Path("/home/pavel/data/goal-misgen/icml-evals")
DEFAULT_TABLE_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/tables"
)

PANELS = (
    {
        "save_name": "coinrun-main-normalized.pdf",
        "auc_tex": "auc-coinrun.tex",
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
        "show_ylabel": True,
    },
    {
        "save_name": "maze-main-normalized.pdf",
        "auc_tex": "auc-maze.tex",
        "prefix": [
            "tmlr-robust-maze-1",
            "tmlr-robust-maze-3",
            "tmlr-oracle-lb-robust400",
            "tmlr-maze-svdd-fix",
        ],
        "env": "maze",
        "robust_filter": "robust",
        "show_ylabel": False,
    },
    {
        "save_name": "heist-main-normalized.pdf",
        "auc_tex": "auc-kandc.tex",
        "prefix": ["tmlr-heist03-expert400"],
        "env": "heist",
        "robust_filter": "all",
        "show_ylabel": False,
    },
)

COMBINED_AUC_TEX = "auc-results.tex"
SHARED_METHOD_FILTER = ["ensemble", "wait"]
PANEL_FIGSIZE = (8, 5.5)
# Large enough that 0.33\linewidth inclusion still reads as paper-size text.
PANEL_AXIS_LABEL_SIZE = 28
PANEL_TICK_LABEL_SIZE = 22
LEGEND_NAME = "main-legend.pdf"

LATEX_SNIPPET = r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{img/main-legend.pdf}
    \vspace{-1.0em}
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
        \caption{\texttt{K\&C}.}
        \label{fig:results:kandc}
    \end{subfigure}
    \caption{Return versus ask-for-help percentage (AFHP), normalized so the
    novice is 0 and the expert is 1. Shaded bands are the interquartile range
    across seeds. The legend applies to all three panels.}
    \label{fig:results}
\end{figure}
""".strip()

LATEX_TABLE_SNIPPET = r"""
\begin{table}[t]
    \centering
    \caption{Area under the return--AFHP curve. Cells are median [IQR] across four seeds.}
    \label{tab:auc_results}
    \input{tables/auc-results}
\end{table}
""".strip()


def _plot_panel(eval_dir: Path, out_dir: Path, table_dir: Path, panel: dict) -> None:
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
        calculate_auc=True,
        robust_filter=panel["robust_filter"],
        normalize_y=True,
        show_legend=False,
        show_ylabel=panel["show_ylabel"],
        figsize=PANEL_FIGSIZE,
        axis_label_size=PANEL_AXIS_LABEL_SIZE,
        tick_label_size=PANEL_TICK_LABEL_SIZE,
        print_auc=False,
        auc_table_path=str(table_dir / panel["auc_tex"]),
    )
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write coinrun/maze/heist main AFHP panels without legends, "
            "a two-line shared legend PDF, per-env AUC tabulars, and one "
            "combined Method × environment AUC table."
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
    parser.add_argument(
        "--table-dir",
        "--table_dir",
        dest="table_dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=(
            "Directory for standalone AUC tabular .tex files "
            f"(default: {DEFAULT_TABLE_DIR})"
        ),
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.is_dir():
        parser.error(f"eval_dir does not exist: {eval_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.table_dir
    table_dir.mkdir(parents=True, exist_ok=True)

    for panel in PANELS:
        _plot_panel(eval_dir, out_dir, table_dir, panel)

    write_auc_env_columns_table(
        table_dir / COMBINED_AUC_TEX,
        [
            (header, table_dir / panel["auc_tex"])
            for header, panel in zip(AUC_ENV_COLUMN_HEADERS, PANELS)
        ],
    )

    save_shared_legend(str(out_dir / LEGEND_NAME), paper_mode=True)

    print("\nLaTeX:\n")
    print(LATEX_SNIPPET)
    print("\n")
    print(LATEX_TABLE_SNIPPET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

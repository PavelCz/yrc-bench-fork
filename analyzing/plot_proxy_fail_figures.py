#!/usr/bin/env python3
"""Write normalized proxy-fail AFHP panels, a shared legend, and AUC tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from analyzing.paper_plot import plot_icml_results, save_shared_legend

matplotlib.use("Agg", force=True)


DEFAULT_EVAL_DIR = Path("/home/pavel/data/goal-misgen/icml-evals")
DEFAULT_TABLE_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/tables"
)
DEFAULT_OUT_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/img"
)

PANELS = (
    {
        "save_name": "coinrun-proxy-fail-normalized.pdf",
        "auc_tex": "auc-coinrun-proxy-fail.tex",
        "prefix": [
            "tmlr-proxy-fail1",
            "tmlr-proxy-fail2",
            "tmlr-proxy-fail3",
            "tmlr-proxy-fail4",
            "tmlr-oracle-lb-proxy_fail",
        ],
        "env": "coinrun_proxy_fail",
        "robust_filter": "all",
        "show_ylabel": True,
        "placeholder": False,
    },
    {
        "save_name": "maze-proxy-fail-normalized.pdf",
        "auc_tex": "auc-maze-proxy-fail.tex",
        "prefix": [
            "tmlr-proxy-fail4",
            "tmlr-oracle-lb-proxy_fail",
            "tmlr-proxy-fail-fix",
        ],
        "env": "maze_proxy_fail",
        "robust_filter": "robust",
        "show_ylabel": False,
        "placeholder": False,
    },
    {
        "save_name": "heist-proxy-fail-normalized.pdf",
        "auc_tex": "auc-kandc-proxy-fail.tex",
        "prefix": ["tmlr-heist03-expert400-proxy-fail"],
        "env": "heist_proxy_fail",
        "robust_filter": "all",
        "show_ylabel": False,
        "placeholder": True,
    },
)

SHARED_METHOD_FILTER = ["ensemble", "wait"]
PANEL_FIGSIZE = (8, 5.5)
PANEL_AXIS_LABEL_SIZE = 28
PANEL_TICK_LABEL_SIZE = 22
LEGEND_NAME = "proxy-fail-legend.pdf"

PLACEHOLDER_AUC_TABULAR = """\
\\begin{tabular}{ll}
\\toprule
Method & AUC (Median [IQR]) \\\\
\\midrule
\\textsc{PartialOracle} & -- \\\\
\\cmidrule(lr){1-2}
\\textsc{Heuristic} & -- \\\\
\\textsc{Ensemble} & -- \\\\
\\textsc{MaxProb} & -- \\\\
\\textsc{MaxLogit} & -- \\\\
\\textsc{ImageSVDD} & -- \\\\
\\textsc{LatentSVDD} & -- \\\\
\\bottomrule
\\end{tabular}
"""

LATEX_SNIPPET = r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{img/proxy-fail-legend.pdf}
    \vspace{-1.0em}
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/coinrun-proxy-fail-normalized.pdf}
        \caption{\texttt{Coinrun}.}
        \label{fig:results:coinrun-proxy-fail}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/maze-proxy-fail-normalized.pdf}
        \caption{\texttt{Maze}.}
        \label{fig:results:maze-proxy-fail}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/heist-proxy-fail-normalized.pdf}
        \caption{\texttt{K\&C} (forthcoming).}
        \label{fig:results:kandc-proxy-fail}
    \end{subfigure}
    \caption{\textbf{Results when proxy goals cause failure.}
    Return versus ask-for-help percentage (AFHP), normalized so the novice
    is 0 and the expert is 1. Shaded bands are the interquartile range
    across seeds. The legend applies to all three panels.}
    \label{fig:results-proxy-fail}
\end{figure}
""".strip()

LATEX_TABLE_SNIPPET = r"""
\begin{table}[t]
    \centering
    \caption{Area Under the Curve (AUC) for Average Return across Ask-For-Help Percentage (AFHP) when proxy goals cause failure.
    IQR shows 25th--75th percentile range among four independent seeds.}
    \label{tab:auc-proxy-fail}
    \begin{subtable}{0.48\linewidth}
        \centering
        \caption{\coin\ AUC results.}
        \label{tab:auc-coinrun-proxy-fail}
        \input{tables/auc-coinrun-proxy-fail}
    \end{subtable}
    \hfill
    \begin{subtable}{0.48\linewidth}
        \centering
        \caption{\maze\ AUC results.}
        \label{tab:auc-maze-proxy-fail}
        \input{tables/auc-maze-proxy-fail}
    \end{subtable}

    \vspace{0.8em}

    \begin{subtable}{0.48\linewidth}
        \centering
        \caption{\kandc\ AUC results (forthcoming).}
        \label{tab:auc-kandc-proxy-fail}
        \input{tables/auc-kandc-proxy-fail}
    \end{subtable}
\end{table}
""".strip()


def _write_placeholder_panel(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Ask-For-Help Percentage (AFHP)", fontsize=PANEL_AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=PANEL_TICK_LABEL_SIZE)
    ax.text(0.5, 0.5, "forthcoming", ha="center", va="center", fontsize=22)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_panel(eval_dir: Path, out_dir: Path, table_dir: Path, panel: dict) -> None:
    save_path = out_dir / panel["save_name"]
    table_path = table_dir / panel["auc_tex"]
    if panel["placeholder"]:
        _write_placeholder_panel(save_path)
        table_path.write_text(PLACEHOLDER_AUC_TABULAR)
        print(f"Wrote placeholder panel to {save_path}")
        print(f"Wrote placeholder AUC table to {table_path}")
        return

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
        auc_table_path=str(table_path),
    )
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write coinrun/maze proxy-fail AFHP panels without legends, "
            "a two-line shared legend PDF, a K&C placeholder panel, "
            "and standalone AUC tabular .tex files."
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
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the PDFs (default: {DEFAULT_OUT_DIR})",
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

    save_shared_legend(str(out_dir / LEGEND_NAME), paper_mode=True)

    print("\nLaTeX:\n")
    print(LATEX_SNIPPET)
    print("\n")
    print(LATEX_TABLE_SNIPPET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

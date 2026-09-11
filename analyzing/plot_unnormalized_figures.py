#!/usr/bin/env python3
"""Write unnormalized AFHP panels for the appendix (main + proxy-fail)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from analyzing.paper_plot import plot_icml_results, save_shared_legend

matplotlib.use("Agg", force=True)


DEFAULT_EVAL_DIR = Path("/home/pavel/data/goal-misgen/icml-evals")
DEFAULT_OUT_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/img"
)

# Coinrun and Maze share the `--paper-app` raw-return window. K&C uses its
# own scale (returns sit near 3.3).
COINRUN_MAZE_YLIM = (6.0, 10.0)

MAIN_PANELS = (
    {
        "save_name": "coinrun-main.pdf",
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
        "ylim": COINRUN_MAZE_YLIM,
        "placeholder": False,
    },
    {
        "save_name": "maze-main.pdf",
        "prefix": [
            "tmlr-robust-maze-1",
            "tmlr-robust-maze-3",
            "tmlr-oracle-lb-robust400",
            "tmlr-maze-svdd-fix",
        ],
        "env": "maze",
        "robust_filter": "robust",
        "show_ylabel": False,
        "ylim": COINRUN_MAZE_YLIM,
        "placeholder": False,
    },
    {
        "save_name": "heist-main.pdf",
        "prefix": ["tmlr-heist03-expert400"],
        "env": "heist",
        "robust_filter": "all",
        "show_ylabel": False,
        "ylim": None,
        "placeholder": False,
    },
)

PROXY_FAIL_PANELS = (
    {
        "save_name": "coinrun-proxy-fail.pdf",
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
        "ylim": COINRUN_MAZE_YLIM,
        "placeholder": False,
    },
    {
        "save_name": "maze-proxy-fail.pdf",
        "prefix": [
            "tmlr-proxy-fail4",
            "tmlr-oracle-lb-proxy_fail",
            "tmlr-proxy-fail-fix",
        ],
        "env": "maze_proxy_fail",
        "robust_filter": "robust",
        "show_ylabel": False,
        "ylim": COINRUN_MAZE_YLIM,
        "placeholder": False,
    },
    {
        "save_name": "heist-proxy-fail.pdf",
        "prefix": ["tmlr-heist03-expert400-proxy-fail"],
        "env": "heist_proxy_fail",
        "robust_filter": "all",
        "show_ylabel": False,
        "ylim": None,
        "placeholder": True,
    },
)

SHARED_METHOD_FILTER = ["ensemble", "wait"]
PANEL_FIGSIZE = (8, 6)
PANEL_AXIS_LABEL_SIZE = 28
PANEL_TICK_LABEL_SIZE = 22
LEGEND_NAME = "unnormalized-legend.pdf"

LATEX_SNIPPET = r"""
\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{img/unnormalized-legend.pdf}
    \vspace{-1.0em}
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/coinrun-main.pdf}
        \caption{\texttt{Coinrun}.}
        \label{fig:app:coinrun-unnormalized}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/maze-main.pdf}
        \caption{\texttt{Maze}.}
        \label{fig:app:maze-unnormalized}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/heist-main.pdf}
        \caption{\texttt{K\&C}.}
        \label{fig:app:kandc-unnormalized}
    \end{subfigure}

    \vspace{0.6em}

    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/coinrun-proxy-fail.pdf}
        \caption{\texttt{Coinrun}, proxy fail.}
        \label{fig:app:coinrun-proxy-fail-unnormalized}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/maze-proxy-fail.pdf}
        \caption{\texttt{Maze}, proxy fail.}
        \label{fig:app:maze-proxy-fail-unnormalized}
    \end{subfigure}%
    \begin{subfigure}{0.33\linewidth}
        \includegraphics[width=\linewidth]{img/heist-proxy-fail.pdf}
        \caption{\texttt{K\&C}, proxy fail (forthcoming).}
        \label{fig:app:kandc-proxy-fail-unnormalized}
    \end{subfigure}
    \caption{\textbf{Unnormalized mean returns.} Top row: recoverable
    environments. Bottom row: proxy pursuit terminates the episode.
    \texttt{Coinrun} and \texttt{Maze} share a $[6, 10]$ y-axis;
    \texttt{K\&C} uses its own scale. Shaded bands are the
    interquartile range across seeds. The legend applies to all panels.}
    \label{fig:results_initial}
\end{figure}
""".strip()


def _write_placeholder_panel(save_path: Path, panel: dict) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.set_xlim(0.0, 1.0)
    if panel["ylim"] is not None:
        ax.set_ylim(*panel["ylim"])
    ax.set_xlabel("Ask-For-Help Percentage (AFHP)", fontsize=PANEL_AXIS_LABEL_SIZE)
    if panel["show_ylabel"]:
        ax.set_ylabel("Average Return", fontsize=PANEL_AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=PANEL_TICK_LABEL_SIZE)
    ax.text(0.5, 0.5, "forthcoming", ha="center", va="center", fontsize=22)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote placeholder panel to {save_path}")


def _plot_panel(eval_dir: Path, out_dir: Path, panel: dict) -> None:
    save_path = out_dir / panel["save_name"]
    if panel["placeholder"]:
        _write_placeholder_panel(save_path, panel)
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
        paper_app=panel["ylim"] is not None,
        calculate_auc=False,
        robust_filter=panel["robust_filter"],
        normalize_y=False,
        show_legend=False,
        show_ylabel=panel["show_ylabel"],
        figsize=PANEL_FIGSIZE,
        axis_label_size=PANEL_AXIS_LABEL_SIZE,
        tick_label_size=PANEL_TICK_LABEL_SIZE,
        ylim=panel["ylim"],
    )
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write unnormalized coinrun/maze/heist AFHP panels for the "
            "appendix (recoverable + proxy-fail). Coinrun and Maze share "
            "a [6, 10] y-axis; K&C autoscales. The K&C proxy-fail panel "
            "is a placeholder."
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
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.is_dir():
        parser.error(f"eval_dir does not exist: {eval_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for panel in (*MAIN_PANELS, *PROXY_FAIL_PANELS):
        _plot_panel(eval_dir, out_dir, panel)

    save_shared_legend(str(out_dir / LEGEND_NAME), paper_mode=True)

    print("\nLaTeX:\n")
    print(LATEX_SNIPPET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write the paper's novice episode-length histograms for seeds 0--7."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from analyzing.episode_length_dist import plot_episode_length_distribution
from analyzing.paper_plot import PAPER_EXP_IDS


DEFAULT_EVAL_DIR = Path("/home/pavel/data/goal-misgen/icml-evals")
DEFAULT_OUT_DIR = Path(
    "/media/linux-data/code/goal-misgen/2025-07-goal-misgeneralization/img"
)

PANELS = (
    {
        "env": "coinrun",
        "prefix": [
            "tmlr2-svdd",
            "tmlr3-svdd",
            "tmlr4-svdd",
            "tmlr5-svdd",
            "imcl04",
            "icml04",
            "tmlr-oracle-lb",
        ],
        "method": "oracle-lb-random",
        "save_name": "coinrun-length-dist.pdf",
        "show_ylabel": True,
    },
    {
        "env": "maze",
        "prefix": [
            "tmlr-robust-maze-1",
            "tmlr-robust-maze-3",
            "tmlr-oracle-lb-robust400",
            "tmlr-maze-svdd-fix",
        ],
        "method": "oracle-lb-random_robust400",
        "save_name": "maze-length-dist.pdf",
        "show_ylabel": False,
    },
    {
        "env": "heist",
        "prefix": ["tmlr-heist03-expert400"],
        "method": "max-prob",
        "save_name": "heist-length-dist.pdf",
        "show_ylabel": False,
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        parser.error(f"eval_dir does not exist: {args.eval_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for panel in PANELS:
        plot_episode_length_distribution(
            eval_dir=args.eval_dir,
            prefix_filter=panel["prefix"],
            env_filter=panel["env"],
            method=panel["method"],
            save_path=str(args.out_dir / panel["save_name"]),
            paper_mode=True,
            show_ylabel=panel["show_ylabel"],
            exp_ids=PAPER_EXP_IDS,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

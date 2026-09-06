#!/usr/bin/env python3
"""Build an improvement-ranked oracle table from two policy_eval JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from YRC.policies.improvement_oracle import (
    save_improvement_table,
    table_from_policy_eval_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair weak and strong policy_eval_results.json files by level seed "
            "and write strong-minus-weak improvements for "
            "ImprovementRankedOraclePolicy."
        )
    )
    parser.add_argument(
        "--weak-json",
        required=True,
        type=Path,
        help="policy_eval_results.json from the novice / weak agent",
    )
    parser.add_argument(
        "--strong-json",
        required=True,
        type=Path,
        help="policy_eval_results.json from the expert / strong agent",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the improvement table JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = table_from_policy_eval_files(args.weak_json, args.strong_json)
    save_improvement_table(
        args.output,
        table["improvements"],
        weak_returns=table["weak_returns"],
        strong_returns=table["strong_returns"],
    )
    print(f"Wrote {len(table['improvements'])} seeds to {args.output}")


if __name__ == "__main__":
    main()

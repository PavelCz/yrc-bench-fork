#!/usr/bin/env python3
"""Print an acting-policy checkpoint path.

Slurm jobs that are queued before training finishes should call this at
start time (after `afterok`) so they pick the final checkpoint instead of
whatever partial `model_*.pth` existed at submit time.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    EXPECTED_TIMESTEPS,
    HEIST400_CHECKPOINT_STEPS,
    get_checkpoint_at_steps,
    get_ensemble_run_dir,
    get_heist400_run_dir,
    get_icml_run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve an acting-policy checkpoint path after training"
    )
    parser.add_argument("--checkpoint-base", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--exp-id", type=int, required=True)
    parser.add_argument("--percent", type=int, choices=[0, 50], default=0)
    parser.add_argument(
        "--heist400",
        action="store_true",
        help="Look up the 400M Heist expert under policy/heist400.",
    )
    parser.add_argument(
        "--ensemble-member",
        type=int,
        default=None,
        help="Look up ensemble member mN under the eval ensembles tree.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Extra parent run directories to search (repeatable).",
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help="Override the expected checkpoint timestep.",
    )
    return parser.parse_args()


def candidate_run_dirs(args: argparse.Namespace) -> list:
    dirs = [Path(run_dir) for run_dir in args.run_dir]
    if args.heist400:
        dirs.append(get_heist400_run_dir(args.exp_id, args.checkpoint_base))
    elif args.ensemble_member is not None:
        dirs.append(
            get_ensemble_run_dir(
                args.env, args.exp_id, args.ensemble_member, args.checkpoint_base
            )
        )
    else:
        dirs.append(
            get_icml_run_dir(args.env, args.exp_id, args.checkpoint_base, args.percent)
        )
    return dirs


def expected_steps_for(args: argparse.Namespace) -> int:
    if args.expected_steps is not None:
        return args.expected_steps
    if args.heist400:
        return HEIST400_CHECKPOINT_STEPS
    return EXPECTED_TIMESTEPS


def main() -> int:
    args = parse_args()
    steps = expected_steps_for(args)
    tried = []
    for run_dir in candidate_run_dirs(args):
        path = Path(get_checkpoint_at_steps(run_dir, steps))
        tried.append(str(path))
        if path.is_file():
            print(path)
            return 0

    print("Checkpoint not found. Tried:", file=sys.stderr)
    for path in tried:
        print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

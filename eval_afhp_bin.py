"""
Single-bin AFHP evaluation worker for SLURM parallel runs.

Evaluates one bin of the AFHP space given a pre-computed calibration state.
Designed to run as a SLURM array job alongside a calibration job submitted by
scripts/run_eval.py.

Typical workflow:
  # 1. Calibration job (run once per experiment)
  python eval_afhp.py --calibrate_only --calibration_path /shared/calib.npz [...]

  # 2. Bin array job (one task per bin, depends on calibration job)
  python eval_afhp_bin.py --bin_idx $SLURM_ARRAY_TASK_ID \\
      --checkpoint_path /shared/results_bin_$SLURM_ARRAY_TASK_ID.npz \\
      --calibration_path /shared/calib.npz [same policy/config args as step 1]

Restart: if --checkpoint_path already exists the job exits immediately, so
re-submitting failed array tasks is safe.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.coverage.coverage_search import load_calibration_state, run_bin
from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy


def load_level_seeds(config) -> Optional[List[int]]:
    """Load ood_eval level seeds from file if configured."""
    level_seeds_file = getattr(config.environment, "level_seeds_file", None)
    if level_seeds_file is None:
        return None

    print(f"Loading level seeds from {level_seeds_file}...")
    with open(level_seeds_file) as f:
        seeds_data = json.load(f)

    level_seeds = seeds_data["seeds"].get("ood_eval", None)
    if level_seeds:
        print(f"  Loaded {len(level_seeds)} ood_eval seeds")
    else:
        print("  No ood_eval seeds in file")

    return level_seeds


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    start_time = time.time()

    # Validate required args
    if args.bin_idx is None:
        raise ValueError("--bin_idx is required")
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required")
    if args.calibration_path is None:
        raise ValueError("--calibration_path is required")

    bin_idx: int = args.bin_idx
    checkpoint_path = Path(args.checkpoint_path)
    calibration_path = Path(args.calibration_path)
    num_bins: int = config.evaluation.num_bins
    afhp_metric: str = config.evaluation.afhp_metric
    search_depth_limit: int = getattr(config.evaluation, "search_depth_limit", 10)

    # Restart support: skip if checkpoint already exists
    if checkpoint_path.exists():
        print(f"Checkpoint already exists, skipping: {checkpoint_path}")
        return

    bin_lo = bin_idx / num_bins
    bin_hi = (bin_idx + 1) / num_bins
    print(
        f"Bin {bin_idx}/{num_bins}: AFHP range [{bin_lo:.3f}, {bin_hi:.3f}], "
        f"metric={afhp_metric}"
    )

    # Load policy (same as eval_afhp.py)
    level_seeds = load_level_seeds(config)

    def make_envs():
        return env_factory.make(config, level_seeds, "sequential")

    envs = make_envs()
    policy = policy_factory.make(config, envs["train"])

    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        algorithms = [
            "timestep_random",
            "level_based_random",
            "threshold",
            "heuristic",
            "wait",
        ]
        if config.general.algorithm not in algorithms:
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config, config.environment)

    # Load calibration state saved by the calibration job
    print(f"Loading calibration state from: {calibration_path}")
    load_calibration_state(policy, calibration_path)

    # Close initial environments; run_bin creates fresh ones per evaluation
    for split_name in envs:
        envs[split_name].close()

    split = "test"
    afhp, performance, meta, threshold = run_bin(
        bin_idx=bin_idx,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
        policy=policy,
        evaluator=evaluator,
        envs_factory=make_envs,
        split=split,
        afhp_metric=afhp_metric,
        search_depth_limit=search_depth_limit,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_path,
        afhp=np.array([afhp]),
        performance=np.array([performance]),
        desired_percentile=np.array([(bin_lo + bin_hi) / 2.0]),
        meta=np.array([meta], dtype=object),
        threshold=np.array([threshold]),
    )

    print(
        f"Bin {bin_idx}: afhp={afhp:.3f}, performance={performance:.4f}, "
        f"threshold={threshold:.4f}"
    )
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Time taken: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

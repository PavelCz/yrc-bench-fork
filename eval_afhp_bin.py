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

import time
from pathlib import Path

import numpy as np

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.eval_setup import build_eval_runtime
from YRC.coverage.coverage_search import load_calibration_state, run_bin


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

    runtime = build_eval_runtime(config)

    # Load calibration state saved by the calibration job
    print(f"Loading calibration state from: {calibration_path}")
    load_calibration_state(runtime.policy, calibration_path)

    # Close initial environments; run_bin creates fresh ones per evaluation
    runtime.close_envs()

    split = "test"
    afhp, performance, meta, threshold = run_bin(
        bin_idx=bin_idx,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
        policy=runtime.policy,
        evaluator=runtime.evaluator,
        envs_factory=runtime.make_envs,
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

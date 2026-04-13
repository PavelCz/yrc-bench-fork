import time

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.artifacts import resolve_calibration_path
from YRC.core.eval_calibration import calibrate_percentile_mapping
from YRC.core.eval_setup import build_eval_runtime
from YRC.coverage.coverage_search import save_calibration_state
from pathlib import Path


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    if args.calibration_path is not None:
        calibration_path = Path(args.calibration_path)
    elif args.coordination_artifact_dir is not None:
        calibration_path = resolve_calibration_path(Path(args.coordination_artifact_dir))
    else:
        raise ValueError(
            "Either --calibration_path or --coordination_artifact_dir is required"
        )

    start_time = time.time()
    runtime = build_eval_runtime(config)

    calibrate_percentile_mapping(
        runtime.policy,
        config,
        runtime.evaluator,
        runtime.envs,
        runtime.make_envs,
        runtime.cal_seeds,
    )

    runtime.close_envs()
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_state(runtime.policy, calibration_path, config)
    print(f"Calibration state saved to: {calibration_path}")
    print(f"Time taken: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

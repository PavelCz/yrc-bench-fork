from pathlib import Path
import time

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.eval_setup import build_eval_runtime
from YRC.coverage.coverage_search import load_calibration_state, run_parallel_eval

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    if args.calibration_path is None:
        raise ValueError("--calibration_path is required")

    start_time = time.time()
    calibration_path = Path(args.calibration_path)
    runtime = build_eval_runtime(config)

    print(f"Loading calibration state from: {calibration_path}")
    load_calibration_state(runtime.policy, calibration_path)

    num_bins: int = config.evaluation.num_bins
    afhp_metric: str = config.evaluation.afhp_metric
    if afhp_metric not in ("level_afhp", "step_afhp"):
        raise ValueError(f"Invalid afhp_metric: {afhp_metric}")

    save_dir = Path(str(get_global_variable("experiment_dir")))
    wandb_kwargs = {
        "name": config.exp_name,
        "project": config.wandb.project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "job_type": "train",
        "config": config,
    }
    if config.wandb.entity is not None:
        wandb_kwargs["entity"] = config.wandb.entity
    exp = wandb.init(**wandb_kwargs)
    wandb_logger = WandbLogger(save_dir=save_dir, experiment=exp)

    split = "test"
    runtime.close_envs()

    log_file_path = get_global_variable("log_file")
    if log_file_path is None:
        raise ValueError(
            "Log file path is not set. Could not find path to save results."
        )
    log_file_path = Path(log_file_path)
    results_file_path = log_file_path.with_name(
        log_file_path.name.replace(".log", f"_{split}.npz")
    )

    print(
        f"Running sequential AFHP eval: num_bins={num_bins}, "
        f"afhp_metric={afhp_metric}"
    )
    results = run_parallel_eval(
        policy=runtime.policy,
        evaluator=runtime.evaluator,
        envs_factory=runtime.make_envs,
        split=split,
        num_bins=num_bins,
        results_path=results_file_path,
        wandb_run=exp,
        logger=wandb_logger,
        afhp_metric=afhp_metric,
    )

    np.savez(
        results_file_path,
        afhps=np.array([r["afhp"] for r in results]),
        performances=np.array([r["performance"] for r in results]),
        desired_percentiles=np.array([r["desired_percentile"] for r in results]),
        meta=np.array([r["meta"] for r in results], dtype=object),
        order=np.array([r["order"] for r in results]),
    )

    print(f"Time taken: {time.time() - start_time} seconds")
    print(f"Total evals: {len(results)}")


if __name__ == "__main__":
    main()

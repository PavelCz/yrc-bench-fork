from pathlib import Path
import time

import flags
import YRC.core.configs.utils as config_utils
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.eval_setup import build_eval_runtime

from YRC.coverage.coverage_search import run_parallel_eval, save_calibration_state

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb


def calibrate_percentile_mapping(policy, config, evaluator, envs, make_envs, cal_seeds):
    """Calibrate the policy's train_percentile_step/level methods.

    Uses fixed validation seeds (held out from RL training, same ID distribution)
    so calibration is reproducible across runs and independent of num_rollouts.

    Different policy types need different calibration:

    - ThresholdPolicy / OODPolicy: Runs rollouts on the "cal" env to collect
      per-step OOD scores and per-episode max scores. Stored in
      policy._train_scores and policy._train_episode_max_scores.

    - TimestepRandomPolicy / ExponentialHeuristicPolicy: Runs the weak agent alone
      on "cal" levels to measure mean episode length (needed because the mapping
      from per-step probability to per-episode help rate is nonlinear).

    - WaitPolicy: Runs the weak agent alone to collect the full distribution of
      episode lengths. train_percentile_level uses empirical percentiles.

    Args:
        policy: The coordination policy to calibrate.
        config: Experiment configuration.
        evaluator: Evaluator instance for running episodes.
        envs: Pre-created environments dict (must include "cal" if cal_seeds given).
        make_envs: Factory that creates fresh environments for calibration runs.
        cal_seeds: Fixed validation seeds used for the "cal" environment, or None
            to fall back to the train env with config.algorithm.num_rollouts.
    """
    from YRC.policies.threshold import ThresholdPolicy
    from YRC.policies.ood import OODPolicy
    from YRC.policies.base import TimestepRandomPolicy
    from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy

    if cal_seeds is None or "cal" not in envs:
        raise ValueError(
            "Calibration requires validation seeds. "
            "Ensure the seed file contains a non-empty 'validation' set and "
            "--level_seeds_file is set."
        )

    num_cal = len(cal_seeds)

    # Score-based calibration: collect OOD score distributions via rollouts
    if isinstance(policy, ThresholdPolicy):
        metric = config.coord_policy.metric
        if metric in ("max_prob", "max_logit", "ensemble_variance"):
            print(
                f"Generating {num_cal} calibration scores for threshold "
                f"policy with {metric} metric..."
            )
            policy.generate_scores(envs["cal"], num_cal)

    if isinstance(policy, OODPolicy) and not isinstance(policy, ThresholdPolicy):
        print(
            f"Generating {num_cal} calibration scores for "
            f"{type(policy).__name__}..."
        )
        policy.generate_scores(envs["cal"], num_cal)

    # Episode-length calibration: run weak agent alone on cal levels
    if isinstance(policy, (TimestepRandomPolicy, ExponentialHeuristicPolicy, WaitPolicy)):
        print(f"Calibrating {type(policy).__name__}: measuring episode lengths...")
        if isinstance(policy, TimestepRandomPolicy):
            old_prob = policy.prob
            policy.prob = 0.0  # weak agent only
        elif isinstance(policy, ExponentialHeuristicPolicy):
            old_prob = 1 - policy.non_ood_starting_prob
            policy.non_ood_starting_prob = 1.0  # weak agent only (ood_starting_prob=0)
        elif isinstance(policy, WaitPolicy):
            old_threshold = policy.threshold
            policy.threshold = 10000  # weak agent only (never ask)
        cal_envs = make_envs()
        cal_summary = evaluator.eval(policy, cal_envs, ["cal"], num_episodes=num_cal, close_envs=True)
        mean_ep_length = cal_summary["cal"]["episode_length_mean"]
        if isinstance(policy, TimestepRandomPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.prob = old_prob
        elif isinstance(policy, ExponentialHeuristicPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.non_ood_starting_prob = 1 - old_prob
        elif isinstance(policy, WaitPolicy):
            policy._episode_lengths = np.array(cal_summary["cal"]["episode_lengths"])
            policy.threshold = old_threshold
        print(f"Mean episode length (weak only): {mean_ep_length:.1f}")


def main():
    args = flags.make()
    args.eval_mode = True
    # Note: config_utils.load() handles logging configuration based on args.log_level
    config = config_utils.load(args.config, flags=args)

    # Record time for profiling purposes
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

    # Calibrate-only mode: save state to disk and exit without running evaluation.
    # Used by the SLURM parallel-bin workflow where bin jobs load the saved state.
    if args.calibrate_only:
        if args.calibration_path is None:
            raise ValueError("--calibration_path is required with --calibrate_only")
        runtime.close_envs()
        save_calibration_state(runtime.policy, args.calibration_path)
        print(f"Calibration state saved to: {args.calibration_path}")
        print(f"Time taken: {time.time() - start_time:.1f}s")
        return

    num_bins: int = config.evaluation.num_bins
    afhp_metric: str = config.evaluation.afhp_metric

    if afhp_metric not in ("level_afhp", "step_afhp"):
        raise ValueError(f"Invalid afhp_metric: {afhp_metric}")

    # Initialize wandb logger
    save_dir = Path(str(get_global_variable("experiment_dir")))

    # Prepare wandb init parameters
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

    wandb_logger = WandbLogger(
        save_dir=save_dir,
        experiment=exp,
    )

    split = "test"

    # Close initial environments - workers create fresh ones for each evaluation
    runtime.close_envs()

    # Save result summary to file.
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
        f"Running parallel bin-based eval: num_bins={num_bins}, "
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

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {len(results)}")


if __name__ == "__main__":
    main()

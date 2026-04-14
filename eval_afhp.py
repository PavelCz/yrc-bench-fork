from pathlib import Path
import json
import os
import time
from typing import List

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy

from YRC.coverage.coverage_search import create_step_afhp_threshold_sampler
from YRC.coverage.coverage_search import create_level_afhp_threshold_sampler

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from acs.types import CurvePoint


def load_level_seeds(config) -> dict:
    """Load evaluation and calibration seeds from file, if configured."""
    level_seeds_file = getattr(config.environment, "level_seeds_file", None)
    if level_seeds_file is None:
        return {"ood_eval": None, "validation": None}

    print(f"Loading level seeds from {level_seeds_file}...")
    with open(level_seeds_file) as f:
        seeds_data = json.load(f)

    ood_eval = seeds_data["seeds"].get("ood_eval") or None
    validation = seeds_data["seeds"].get("validation") or None

    if ood_eval:
        print(f"  - Loaded {len(ood_eval)} ood_eval seeds")
    if validation:
        print(f"  - Loaded {len(validation)} validation seeds (calibration)")

    return {"ood_eval": ood_eval, "validation": validation}


def _require_calibration_split_and_count(envs, cal_seeds):
    """Resolve the fixed-seed calibration split and episode count."""
    if cal_seeds is None or "cal" not in envs:
        raise ValueError(
            "Calibration requires validation seeds. Ensure the seed file contains "
            "a non-empty 'validation' set and --level_seeds_file is set."
        )
    return "cal", len(cal_seeds)


def calibrate_percentile_mapping(policy, config, evaluator, envs, make_envs, cal_seeds):
    """Calibrate the policy's train_percentile_step/level methods.

    Different policy types need different calibration:

    - ThresholdPolicy / OODPolicy: Runs rollouts to collect per-step OOD scores and
      per-episode max scores. These are stored in policy._train_scores and
      policy._train_episode_max_scores for use by train_percentile_step/level.

    - TimestepRandomPolicy / ExponentialHeuristicPolicy: Runs the weak agent alone
      (no help requests) on calibration levels to measure mean episode length. This
      is needed because the mapping from per-step probability to per-episode help
      rate is nonlinear (see docs/percentile_calibration.md).

    - WaitPolicy: Runs the weak agent alone to collect the full distribution of
      episode lengths. train_percentile_level uses empirical percentiles of this
      distribution, since an episode has help iff its length exceeds the wait
      threshold.

    Args:
        policy: The coordination policy to calibrate.
        config: Experiment configuration.
        evaluator: Evaluator instance for running episodes.
        envs: Pre-created environments (used for score generation rollouts).
        make_envs: Factory that creates fresh environments for calibration runs.
        cal_seeds: Fixed validation seeds for reproducible calibration.
    """
    from YRC.policies.threshold import ThresholdPolicy
    from YRC.policies.ood import OODPolicy
    from YRC.policies.base import (
        OracleLevelBasedRandomPolicy,
        TimestepRandomPolicy,
    )
    from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy

    if isinstance(policy, OracleLevelBasedRandomPolicy):
        print(
            "Skipping calibration for OracleLevelBasedRandomPolicy; "
            "using the built-in 50% OOD assumption for level AFHP mapping."
        )
        return

    cal_split, num_cal_episodes = _require_calibration_split_and_count(envs, cal_seeds)
    cal_env = envs[cal_split]

    # Score-based calibration: collect OOD score distributions via rollouts
    if isinstance(policy, ThresholdPolicy):
        metric = config.coord_policy.metric
        if metric in ("max_prob", "max_logit", "ensemble_variance"):
            if num_cal_episodes % cal_env.num_envs != 0:
                raise ValueError(
                    "Calibration seed count must be divisible by "
                    f"environment.common.num_envs ({cal_env.num_envs}) for "
                    "score-based calibration."
                )
            print(
                f"Generating {num_cal_episodes} calibration scores for threshold "
                f"policy with {metric} metric..."
            )
            policy.generate_scores(cal_env, num_cal_episodes)

    if isinstance(policy, OODPolicy) and not isinstance(policy, ThresholdPolicy):
        if num_cal_episodes % cal_env.num_envs != 0:
            raise ValueError(
                "Calibration seed count must be divisible by "
                f"environment.common.num_envs ({cal_env.num_envs}) for "
                "score-based calibration."
            )
        print(
            f"Generating {num_cal_episodes} calibration scores for "
            f"{type(policy).__name__}..."
        )
        policy.generate_scores(cal_env, num_cal_episodes)

    # Episode-length calibration: run weak agent alone on calibration levels
    if isinstance(
        policy, (TimestepRandomPolicy, ExponentialHeuristicPolicy, WaitPolicy)
    ):
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
        cal_summary = evaluator.eval(
            policy,
            cal_envs,
            [cal_split],
            num_episodes=num_cal_episodes,
            close_envs=True,
        )
        mean_ep_length = cal_summary[cal_split]["episode_length_mean"]
        if isinstance(policy, TimestepRandomPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.prob = old_prob
        elif isinstance(policy, ExponentialHeuristicPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.non_ood_starting_prob = 1 - old_prob
        elif isinstance(policy, WaitPolicy):
            policy._episode_lengths = np.array(
                cal_summary[cal_split]["episode_lengths"]
            )
            policy.threshold = old_threshold
        print(f"Mean episode length (weak only): {mean_ep_length:.1f}")


def main():
    args = flags.make()
    args.eval_mode = True
    # Note: config_utils.load() handles logging configuration based on args.log_level
    config = config_utils.load(args.config, flags=args)

    # Record time for profiling purposes
    start_time = time.time()

    # Load level seeds for evaluation and optional fixed-seed calibration.
    seeds = load_level_seeds(config)
    level_seeds = seeds["ood_eval"]
    cal_seeds = seeds["validation"]

    # Create environment factory for the sampler
    # Each evaluation gets fresh environments with the same seeds in sequential order
    def make_envs():
        return env_factory.make(
            config,
            level_seeds,
            "sequential",
            cal_seeds=cal_seeds,
        )

    # Create initial environments for policy creation and score generation
    envs = make_envs()

    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        # The following algorithms do not need to load a model, because they do not
        # need the training step:
        algorithms = [
            "timestep_random",
            "level_based_random",
            "oracle_level_based_random",
            "threshold",
            "heuristic",
            "wait",
        ]
        if config.general.algorithm not in algorithms:
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config, config.environment)

    calibrate_percentile_mapping(
        policy,
        config,
        evaluator,
        envs,
        make_envs,
        cal_seeds,
    )

    coverage_fraction = config.evaluation.coverage_fraction
    threshold_sampler: str = config.evaluation.threshold_sampler

    if coverage_fraction < 0.01:
        raise ValueError("Coverage fraction must be at least 0.01")

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

    # Create the joint-coverage sampler via YRC wrapper (adapts to new abcs API)
    max_total_evals = 200

    # Close initial environments - the sampler will create fresh ones for each evaluation
    for split_name in envs:
        envs[split_name].close()

    if threshold_sampler == "step_afhp":
        sampler = create_step_afhp_threshold_sampler(
            policy=policy,
            evaluator=evaluator,
            envs_factory=make_envs,
            split=split,
            coverage_fraction=coverage_fraction,
            max_total_evals=max_total_evals,
            logger=wandb_logger,
            wandb_run=exp,
        )
    elif threshold_sampler == "level_afhp":
        sampler = create_level_afhp_threshold_sampler(
            policy=policy,
            evaluator=evaluator,
            envs_factory=make_envs,
            split=split,
            coverage_fraction=coverage_fraction,
            max_total_evals=max_total_evals,
            logger=wandb_logger,
            wandb_run=exp,
        )
    else:
        raise ValueError(f"Invalid threshold sampler: {threshold_sampler}")

    # Run the sampling
    print(
        f"Running joint coverage sampling with coverage_fraction="
        f"{coverage_fraction:.3f}, budget={max_total_evals}..."
    )
    sampling_result = sampler.run()

    # Report coverage
    print(
        f"Coverage x-gap: {sampling_result.coverage_x_max_gap:.3f}, "
        f"y-gap: {sampling_result.coverage_y_max_gap:.3f}"
    )

    # TODO: Rename
    # The sort metric is called afhp for legacy reasons, sort metric or threshold metric
    # would be more appropriate.
    sorted_points: List[CurvePoint] = sorted(
        sampling_result.points, key=lambda p: p.afhp
    )

    total_evals = sampling_result.total_evals

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
    np.savez(
        results_file_path,
        afhps=np.array([pt.afhp for pt in sorted_points]),
        performances=np.array([pt.performance for pt in sorted_points]),
        desired_percentiles=np.array([pt.desired_percentile for pt in sorted_points]),
        meta=np.array([pt.meta for pt in sorted_points]),
        order=np.array([pt.order for pt in sorted_points]),
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {total_evals}")


if __name__ == "__main__":
    main()

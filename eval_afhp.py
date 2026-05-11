from pathlib import Path
import os
import logging
import time
from typing import List

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.eval_script_utils import init_eval_wandb_run, save_npz_results
from YRC.core.level_seeds import load_level_seed_splits

from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy

from YRC.coverage.coverage_search import create_step_afhp_threshold_sampler
from YRC.coverage.coverage_search import create_level_afhp_threshold_sampler

import numpy as np
from pytorch_lightning.loggers import WandbLogger
from acs.types import CurvePoint


def require_non_plain_maze_eval_env(env_name: str) -> None:
    if env_name == "maze":
        raise ValueError(
            "Plain Procgen env 'maze' does not expose randomize_goal labels. "
            "Use 'maze_afh' for eval_afhp.py maze evaluations."
        )


def _require_calibration_split_and_count(envs, cal_seeds):
    """Resolve the fixed-seed calibration split and episode count."""
    if cal_seeds is None or "cal" not in envs:
        raise ValueError(
            "Calibration requires validation seeds. Ensure the seed file contains "
            "a non-empty 'validation' set and --level_seeds_file is set."
        )
    return "cal", len(cal_seeds)


def _select_calibration_seeds(config, seeds):
    calibration_levels = getattr(config.evaluation, "calibration_levels", None)
    cal_seeds = seeds["validation"]
    if calibration_levels is None:
        return cal_seeds
    if calibration_levels <= 0:
        raise ValueError("evaluation.calibration_levels must be positive when set.")
    if calibration_levels > len(cal_seeds):
        raise ValueError(
            "Requested evaluation.calibration_levels="
            f"{calibration_levels}, but validation split contains only "
            f"{len(cal_seeds)} seeds."
        )
    return cal_seeds[:calibration_levels]


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
        LevelBasedRandomPolicy,
        OracleLevelBasedRandomPolicy,
        TimestepRandomPolicy,
    )
    from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy

    if isinstance(policy, (LevelBasedRandomPolicy, OracleLevelBasedRandomPolicy)):
        message = (
            f"Skipping calibration for {type(policy).__name__}; "
            "using the policy's built-in level AFHP mapping."
        )
        print(message)
        logging.info(message)
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
            message = (
                f"Generating {num_cal_episodes} calibration scores for threshold "
                f"policy with {metric} metric..."
            )
            print(message)
            logging.info(message)
            policy.generate_scores(cal_env, num_cal_episodes)

    if isinstance(policy, OODPolicy) and not isinstance(policy, ThresholdPolicy):
        if num_cal_episodes % cal_env.num_envs != 0:
            raise ValueError(
                "Calibration seed count must be divisible by "
                f"environment.common.num_envs ({cal_env.num_envs}) for "
                "score-based calibration."
            )
        message = (
            f"Generating {num_cal_episodes} calibration scores for "
            f"{type(policy).__name__}..."
        )
        print(message)
        logging.info(message)
        policy.generate_scores(cal_env, num_cal_episodes)

    # Episode-length calibration: run weak agent alone on calibration levels
    if isinstance(
        policy, (TimestepRandomPolicy, ExponentialHeuristicPolicy, WaitPolicy)
    ):
        message = f"Calibrating {type(policy).__name__}: measuring episode lengths..."
        print(message)
        logging.info(message)
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
        message = f"Mean episode length (weak only): {mean_ep_length:.1f}"
        print(message)
        logging.info(message)


def main():
    args = flags.make()
    args.eval_mode = True
    # Note: config_utils.load() handles logging configuration based on args.log_level
    config = config_utils.load(args.config, flags=args)
    require_non_plain_maze_eval_env(config.environment.common.env_name)

    # Record time for profiling purposes
    start_time = time.time()

    # Load all seed splits, then explicitly map semantic seed splits to env splits.
    seeds = load_level_seed_splits(config, required_splits=("ood_eval", "validation"))
    cal_seeds = _select_calibration_seeds(config, seeds)
    level_seeds_by_split = {
        "test": seeds["ood_eval"],
        "cal": cal_seeds,
    }

    # Create environment factory for the sampler
    # Each evaluation gets fresh environments with the same seeds in sequential order
    def make_envs():
        return env_factory.make(
            config,
            level_seeds_by_split=level_seeds_by_split,
            level_seeds_mode="sequential",
            require_level_seeds_for_splits=("test", "cal"),
        )

    # Create initial environments for policy creation and score generation
    logging.info("Creating initial evaluation environments.")
    env_start = time.time()
    envs = make_envs()
    logging.info(
        f"Initial evaluation environments created in {time.time() - env_start:.2f}s"
    )

    logging.info("Creating coordination policy.")
    policy_start = time.time()
    policy = policy_factory.make(config, envs["train"])
    logging.info(f"Coordination policy created in {time.time() - policy_start:.2f}s")
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
            model_path = os.path.join(config.experiment_dir, config.file_name)
            logging.info(f"Loading coordination policy model from {model_path}")
            model_load_start = time.time()
            policy.load_model(model_path)
            logging.info(
                "Coordination policy model loaded in "
                f"{time.time() - model_load_start:.2f}s"
            )

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config, config.environment)

    logging.info("Starting percentile calibration.")
    calibration_start = time.time()
    calibrate_percentile_mapping(
        policy,
        config,
        evaluator,
        envs,
        make_envs,
        cal_seeds,
    )
    logging.info(
        f"Percentile calibration completed in {time.time() - calibration_start:.2f}s"
    )

    coverage_fraction = config.evaluation.coverage_fraction
    threshold_sampler: str = config.evaluation.threshold_sampler

    if coverage_fraction < 0.01:
        raise ValueError("Coverage fraction must be at least 0.01")

    # Initialize wandb logger
    save_dir = Path(str(get_global_variable("experiment_dir")))

    exp = init_eval_wandb_run(
        config,
        name=config.exp_name,
        job_type="train",
        run_config=config,
    )

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
            image_svdd_degenerate_strategy=getattr(
                config.evaluation,
                "image_svdd_degenerate_strategy",
                "expand_above_id",
            ),
            image_svdd_expansion_max_evals=getattr(
                config.evaluation,
                "image_svdd_expansion_max_evals",
                12,
            ),
            image_svdd_expansion_initial_delta_fraction=getattr(
                config.evaluation,
                "image_svdd_expansion_initial_delta_fraction",
                1e-4,
            ),
        )
    else:
        raise ValueError(f"Invalid threshold sampler: {threshold_sampler}")

    # Run the sampling
    print(
        f"Running joint coverage sampling with coverage_fraction="
        f"{coverage_fraction:.3f}, budget={max_total_evals}..."
    )
    logging.info(
        "Running joint coverage sampling with "
        f"coverage_fraction={coverage_fraction:.3f}, budget={max_total_evals}"
    )
    sampling_result = sampler.run()

    # Report coverage
    print(
        f"Coverage x-gap: {sampling_result.coverage_x_max_gap:.3f}, "
        f"y-gap: {sampling_result.coverage_y_max_gap:.3f}"
    )
    if sampling_result.info is not None:
        logging.info("Sampling info: %s", sampling_result.info)
        threshold_strategy = sampling_result.info.get("threshold_strategy")
        if threshold_strategy is not None:
            print(f"Threshold strategy: {threshold_strategy}")

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
    save_npz_results(
        results_file_path,
        afhps=np.array([pt.afhp for pt in sorted_points]),
        performances=np.array([pt.performance for pt in sorted_points]),
        desired_percentiles=np.array([pt.desired_percentile for pt in sorted_points]),
        meta=np.array([pt.meta for pt in sorted_points]),
        order=np.array([pt.order for pt in sorted_points]),
        sampling_info=np.array([sampling_result.info], dtype=object),
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {total_evals}")


if __name__ == "__main__":
    main()

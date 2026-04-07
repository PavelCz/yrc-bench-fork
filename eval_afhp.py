from pathlib import Path
import json
import os
import time
from typing import List, Optional

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy

from YRC.coverage.coverage_search import run_parallel_eval, save_calibration_state

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb



def load_level_seeds(config) -> Optional[List[int]]:
    """Load ood_eval level seeds from file if configured.
    
    Args:
        config: Configuration object with environment.level_seeds_file path
        
    Returns:
        List of level seeds for OOD evaluation, or None if not configured
    """
    level_seeds_file = getattr(config.environment, 'level_seeds_file', None)
    if level_seeds_file is None:
        return None
    
    print(f'LOADING LEVEL SEEDS FROM {level_seeds_file}...')
    with open(level_seeds_file) as f:
        seeds_data = json.load(f)
    
    # Use ood_eval seeds for evaluation (sequential mode, fresh envs per eval)
    level_seeds = seeds_data['seeds'].get('ood_eval', None)
    if level_seeds:
        print(f'  - Loaded {len(level_seeds)} ood_eval seeds (mode: sequential)')
    else:
        print('  - No ood_eval seeds in file')
    
    return level_seeds


def calibrate_percentile_mapping(policy, config, evaluator, envs, make_envs):
    """Calibrate the policy's train_percentile_step/level methods using training data.

    Different policy types need different calibration:

    - ThresholdPolicy / OODPolicy: Runs rollouts to collect per-step OOD scores and
      per-episode max scores. These are stored in policy._train_scores and
      policy._train_episode_max_scores for use by train_percentile_step/level.

    - TimestepRandomPolicy / ExponentialHeuristicPolicy: Runs the weak agent alone
      (no help requests) on training levels to measure mean episode length. This is
      needed because the mapping from per-step probability to per-episode help rate
      is nonlinear (see docs/percentile_calibration.md).

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
    """
    from YRC.policies.threshold import ThresholdPolicy
    from YRC.policies.ood import OODPolicy
    from YRC.policies.base import TimestepRandomPolicy
    from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy

    # Score-based calibration: collect OOD score distributions via rollouts
    if isinstance(policy, ThresholdPolicy):
        metric = config.coord_policy.metric
        if metric in ("max_prob", "max_logit", "ensemble_variance"):
            num_rollouts = getattr(config.algorithm, "num_rollouts", 256)
            print(
                f"Generating {num_rollouts} training scores for threshold "
                f"policy with {metric} metric..."
            )
            policy.generate_scores(envs["train"], num_rollouts)

    if isinstance(policy, OODPolicy) and not isinstance(policy, ThresholdPolicy):
        num_rollouts = getattr(config.algorithm, "num_rollouts", 256)
        print(
            f"Generating {num_rollouts} training scores for "
            f"{type(policy).__name__}..."
        )
        policy.generate_scores(envs["train"], num_rollouts)

    # Episode-length calibration: run weak agent alone on training levels
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
        cal_summary = evaluator.eval(policy, cal_envs, ["train"], close_envs=True)
        mean_ep_length = cal_summary["train"]["episode_length_mean"]
        if isinstance(policy, TimestepRandomPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.prob = old_prob
        elif isinstance(policy, ExponentialHeuristicPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.non_ood_starting_prob = 1 - old_prob
        elif isinstance(policy, WaitPolicy):
            policy._episode_lengths = np.array(
                cal_summary["train"]["episode_lengths"]
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

    # Load level seeds for evaluation
    level_seeds = load_level_seeds(config)

    # Create environment factory for the sampler
    # Each evaluation gets fresh environments with the same seeds in sequential order
    def make_envs():
        return env_factory.make(config, level_seeds, "sequential")

    # Create initial environments for policy creation and score generation
    envs = make_envs()

    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        # The following algorithms do not need to load a model, because they do not
        # need the training step:
        algorithms = ["timestep_random", "level_based_random", "threshold", "heuristic", "wait"]
        if config.general.algorithm not in algorithms:
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config, config.environment)

    calibrate_percentile_mapping(policy, config, evaluator, envs, make_envs)

    # Calibrate-only mode: save state to disk and exit without running evaluation.
    # Used by the SLURM parallel-bin workflow where bin jobs load the saved state.
    if args.calibrate_only:
        if args.calibration_path is None:
            raise ValueError("--calibration_path is required with --calibrate_only")
        for split_name in envs:
            envs[split_name].close()
        save_calibration_state(policy, args.calibration_path)
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
    for split_name in envs:
        envs[split_name].close()

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
        policy=policy,
        evaluator=evaluator,
        envs_factory=make_envs,
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

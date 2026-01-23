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

from YRC.coverage.coverage_search import create_afhp_threshold_sampler
from YRC.coverage.coverage_search import create_ood_percentage_threshold_sampler

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from acs.types import CurvePoint



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

    # For threshold policy with metrics that require training score distribution (like max_logit),
    # we need to generate scores before running AFHP evaluation
    from YRC.policies.threshold import ThresholdPolicy

    if isinstance(policy, ThresholdPolicy):
        metric = config.coord_policy.metric
        # These metrics require the training score distribution for percentile computation
        if metric in ("max_logit", "ensemble_variance"):
            # Use algorithm.num_rollouts if available, otherwise use a default
            num_rollouts = getattr(config.algorithm, "num_rollouts", 256)
            print(
                f"Generating {num_rollouts} training scores for threshold policy with {metric} metric..."
            )
            policy.generate_scores(envs["train"], num_rollouts)

    evaluator = Evaluator(config, config.environment)

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

    if threshold_sampler == "afhp":
        sampler = create_afhp_threshold_sampler(
            policy=policy,
            evaluator=evaluator,
            envs_factory=make_envs,
            split=split,
            coverage_fraction=coverage_fraction,
            max_total_evals=max_total_evals,
            logger=wandb_logger,
        )
    elif threshold_sampler == "ood_percentage":
        sampler = create_ood_percentage_threshold_sampler(
            policy=policy,
            evaluator=evaluator,
            envs_factory=make_envs,
            split=split,
            coverage_fraction=coverage_fraction,
            max_total_evals=max_total_evals,
            logger=wandb_logger,
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

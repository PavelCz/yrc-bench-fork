from pathlib import Path
import os
import time

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

from YRC.policies.lightning_ae import LightningAEPolicy
from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy
from YRC.policies.ood import OODPolicy
from YRC.policies.base import RandomPolicy

from YRC.coverage import create_threshold_sampler

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import Tuple, Dict, Any


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    # Record time for profiling purposes
    start_time = time.time()

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        # If we are doing threshold search, the random alg does not need to train
        # anything. Thus, we do not need to load here.
        if config.general.algorithm != "random":
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config.evaluation)

    coverage_fraction = args.eval.coverage_fraction

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
    coverage_fraction = 1.0 / float(coverage_fraction)
    max_total_evals = 200

    # Collect metadata via callback
    summaries_by_p: Dict[float, Dict[str, Any]] = {}
    thresholds_by_p: Dict[float, float] = {}

    def on_eval(p: float, thr: float, summary: Dict[str, Any]) -> None:
        summaries_by_p[p] = summary
        thresholds_by_p[p] = thr

    sampler = create_threshold_sampler(
        policy=policy,
        evaluator=evaluator,
        envs=envs,
        split=split,
        coverage_fraction=coverage_fraction,
        max_total_evals=max_total_evals,
        logger=wandb_logger,
        on_eval=on_eval,
    )

    # Run the sampling
    print(
        f"Running joint coverage sampling with coverage_fraction={coverage_fraction:.3f}, "
        f"budget={max_total_evals}..."
    )
    sampling_result = sampler.run()

    # Report coverage
    print(
        f"Coverage x-gap: {sampling_result.coverage_x_max_gap:.3f}, "
        f"y-gap: {sampling_result.coverage_y_max_gap:.3f}"
    )

    # Extract summaries, percentiles, and thresholds from points (sorted by percentile)
    summaries = []
    binned_train_percentiles = []
    binned_thresholds = []

    sorted_points = sorted(sampling_result.points, key=lambda p: p.percentile)
    for pt in sorted_points:
        p = float(pt.percentile)
        summaries.append(summaries_by_p.get(p))
        binned_train_percentiles.append(p * 100.0)
        binned_thresholds.append(thresholds_by_p.get(p))

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
        binned_train_percentiles=binned_train_percentiles,
        binned_thresholds=binned_thresholds,
        results=np.array(summaries),
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {total_evals}")


def update_policy_params(policy, threshold):
    if isinstance(policy, LightningAEPolicy) or isinstance(policy, OODPolicy):
        policy.update_params({"threshold": threshold})
    elif isinstance(policy, RandomPolicy):
        if threshold == float("inf"):
            # An infinite threshold means that the policy will never ask for help.
            # We need to set the probability to 0.
            threshold = 0.0
        elif threshold == float("-inf"):
            # A negative infinite threshold means that the policy will always ask for help.
            # We need to set the probability to 1.
            threshold = 1.0
        policy.update_params(threshold)

    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )
    


if __name__ == "__main__":
    main()

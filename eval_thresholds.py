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

from YRC.coverage.binary_search import BinarySearchSampler

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

    num_threshold_bins = args.eval.threshold_bins

    if num_threshold_bins < 5:
        raise ValueError("Number of threshold bins must be at least 5")

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

    # Create evaluation function
    eval_function = create_eval_function(policy, evaluator, envs, split, wandb_logger)

    # Create percentile to threshold converter
    def input_to_threshold(percentile: float) -> float:
        return percentile_to_threshold(policy, percentile)

    # Create the binary search sampler
    sampler = BinarySearchSampler(
        eval_function=eval_function,
        num_bins=num_threshold_bins,
        input_range=(0.0, 100.0),  # Percentiles
        output_range=(0.0, 100.0),  # AFHP percentage
        input_to_threshold=input_to_threshold,
    )

    # Run the sampling
    print(f"Running binary search sampling with {num_threshold_bins} bins...")
    bin_samples = sampler.run()

    # Get coverage summary
    coverage_summary = sampler.get_coverage_summary()
    print(f"Coverage: {coverage_summary['coverage_percentage']:.1f}%")
    print(f"Bins filled: {coverage_summary['bins_filled']}/{num_threshold_bins}")

    # Extract summaries, percentiles, and thresholds from samples
    summaries = []
    binned_train_percentiles = []
    binned_thresholds = []

    for sample in bin_samples:
        if sample is not None:
            summaries.append(sample.metadata["summary"])
            binned_train_percentiles.append(sample.input_value)
            binned_thresholds.append(sample.metadata["threshold"])
        else:
            summaries.append(None)
            binned_train_percentiles.append(None)
            binned_thresholds.append(None)

    total_evals = sampler.total_evals

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


def create_eval_function(policy, evaluator, envs, split, wandb_logger):
    """Create an evaluation function for the BinarySearchSampler."""

    def eval_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        """Evaluate policy at given threshold."""
        # Update policy with threshold
        update_policy_params(policy, threshold)

        # Run evaluation
        summary = evaluator.eval(
            policy, envs, [split], logger=wandb_logger, threshold=threshold
        )

        # Extract AFHP as output value (convert to percentage)
        afhp = summary[split]["action_1_frac"] * 100

        # Return AFHP and full summary as metadata
        return afhp, {"summary": summary, "threshold": threshold}

    return eval_function


def percentile_to_threshold(policy, percentile: float) -> float:
    """Convert percentile to threshold."""
    if percentile == 0:
        return float("inf")
    elif percentile == 100:
        return float("-inf")
    else:
        return policy.train_percentile(100 - percentile)


if __name__ == "__main__":
    main()

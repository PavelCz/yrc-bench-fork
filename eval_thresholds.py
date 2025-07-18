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
from YRC.policies.ood import OODPolicy
from YRC.policies.base import RandomPolicy

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import List


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

    summaries = [None] * num_threshold_bins
    # A list of bins, where each value in the bin is the TRAINING (inverse) percentile,
    # based on the empirical AFHP value.
    # I.e. index one corresponds to the bin 0% <= AFHP <= 10%, and the value inside
    # will be the inverse precentile of the threshold that got binned into that bin.
    binned_train_percentiles = [None] * num_threshold_bins
    binned_thresholds = [None] * num_threshold_bins
    afhp_bins = np.linspace(0, 100, num_threshold_bins + 1)

    # Evaluate for extreme values.
    # A threshold of -inf should correspond with a 100% AFHP.
    update_policy_params(policy, float("-inf"))
    summary = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=float("-inf")
    )
    summaries[-1] = summary
    binned_train_percentiles[-1] = 100
    binned_thresholds[-1] = float("-inf")
    # A threshold of inf should correspond with a 0% AFHP.
    update_policy_params(policy, float("inf"))
    summary = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=float("inf")
    )
    summaries[0] = summary
    binned_train_percentiles[0] = 0
    binned_thresholds[0] = float("inf")

    # These indices point to the actual bins, where the training inverse percentiles got
    # binned to. We want to do a bisecting search, where start at 0,100 and then we use
    # dynamic programming to insert 25 and 75 into the bins and so on.
    left_index = 0
    right_index = num_threshold_bins - 1
    left_percentile = 0
    right_percentile = 100

    total_evals = 2
    new_evals = determine_results(
        summaries,
        binned_train_percentiles,
        binned_thresholds,
        afhp_bins,
        left_percentile,
        right_percentile,
        left_index,
        right_index,
        split,
        policy,
        envs,
        wandb_logger,
        evaluator,
    )
    total_evals += new_evals

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


def determine_results(
    summaries: List[dict],
    binned_train_percentiles: List[float],
    binned_thresholds: List[float],
    afhp_bins: List[float],
    left_percentile: float,
    right_percentile: float,
    left_index: int,
    right_index: int,
    split: str,
    policy,
    envs,
    wandb_logger,
    evaluator,
):
    # Determine the new percentile to check, which will be the middle of the two
    # percentiles indicated by the left and right indices.
    middle_percentile = (left_percentile + right_percentile) / 2

    # Determine the threshold for the given middle percentile. Remember that these are
    # inverse percentiles, so we need to invert the percentile to get the threshold.
    middle_threshold = policy.train_percentile(100 - middle_percentile)

    # Update the policy with the new threshold.
    update_policy_params(policy, middle_threshold)

    # Run eval with the new threshold.
    summary = evaluator.eval(
        policy,
        envs,
        [split],
        logger=wandb_logger,
        threshold=middle_threshold,
    )
    # This is the empirically determined AFHP.
    afhp = summary[split]["action_1_frac"]
    # Determine which bin the AFHP would go into.
    bin_idx = determine_bin(afhp_bins, afhp)

    # We only add new evals to the bin if it is empty.
    if summaries[bin_idx] is None:
        # If the bin is empty, we can add the summary and threshold to the bin.
        summaries[bin_idx] = summary
        binned_thresholds[bin_idx] = middle_threshold
        binned_train_percentiles[bin_idx] = middle_percentile

    # Check if empty bins remain left
    left_remaining = bins_remaining(summaries, left_index, bin_idx)
    evals_left = 0
    if left_remaining:
        evals_left = determine_results(
            summaries,
            binned_train_percentiles,
            binned_thresholds,
            afhp_bins,
            left_percentile,
            middle_percentile,
            left_index,
            bin_idx,
            split,
            policy,
            envs,
            wandb_logger,
            evaluator,
        )
    evals_right = 0
    right_remaining = bins_remaining(summaries, bin_idx, right_index)
    if right_remaining:
        evals_right = determine_results(
            summaries,
            binned_train_percentiles,
            binned_thresholds,
            afhp_bins,
            middle_percentile,
            right_percentile,
            bin_idx,
            right_index,
            split,
            policy,
            envs,
            wandb_logger,
            evaluator,
        )
    return evals_left + evals_right + 1


def determine_bin(afhp_bins: List[float], afhp: float) -> int:
    if afhp < 0.0 or afhp > 1.0:
        raise ValueError(f"Error, encountered an AFHP of {afhp}")
    afhp_percent = afhp * 100
    for i in range(len(afhp_bins) - 1):
        if afhp_percent >= afhp_bins[i] and afhp_percent <= afhp_bins[i + 1]:
            return i
    raise ValueError(
        f"Encountered issue with percentile_bins {afhp_bins} and afhp {afhp_percent}."
    )


def bins_remaining(summaries, left_index, right_index) -> bool:
    for i in range(left_index + 1, right_index):
        if summaries[i] is None:
            # Found at least one remaining bin that wasn't filled yet.
            return True
    return False


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

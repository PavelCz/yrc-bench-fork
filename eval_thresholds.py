from pathlib import Path
import os

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

    # For our initial set of thresolds, we determine threshold percentiles based on
    # training scores
    initial_thresholds, percentile_steps = policy.compute_train_percentiles(3)

    # Linearly extend the thresholds below the lowest threshold.
    # delta = thresholds[-1] - thresholds[0]
    # Similarly, extend the thresholds above the highest threshold.
    # additional_thresholds = []
    # highest_threshold = thresholds[-1]
    # for i in range(0, args.eval.num_thresholds * 2):
    #     additional_thresholds.append(highest_threshold + delta * (2**i))
    # thresholds = np.concatenate([thresholds, np.array(additional_thresholds)])

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
    binned_thresholds = [None] * num_threshold_bins
    percentile_bins = np.linspace(0, 100, num_threshold_bins + 1)

    # Evaluat for extreme values
    update_policy_params(policy, float("-inf"))
    summary = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=float("-inf")
    )
    summaries[0] = summary
    update_policy_params(policy, float("inf"))
    summary = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=float("inf")
    )
    summaries[-1] = summary

    binned_thresholds[0] = initial_thresholds[0]
    binned_thresholds[-1] = initial_thresholds[-1]

    # current_num_evals = 2

    left_index = 0
    right_index = num_threshold_bins - 0
    left_threshold = initial_thresholds[0]
    right_threshold = initial_thresholds[-1]

    update_policy_params(policy, left_threshold)
    summary = evaluator.eval(
        policy,
        envs,
        [split],
        logger=wandb_logger,
        threshold=left_threshold,
        # percentile_step=percentile_step,
    )
    afhp = summary[split]["action_1_frac"]
    if afhp > percentile_bins[1]:
        binned_thresholds[1] = left_threshold
        summaries[1] = summary
        left_index += 1

    update_policy_params(policy, right_threshold)
    summary = evaluator.eval(
        policy,
        envs,
        [split],
        logger=wandb_logger,
        threshold=right_threshold,
        # percentile_step=percentile_step,
    )
    # summary_batch.append(summary)
    afhp = summary[split]["action_1_frac"]
    if afhp < percentile_bins[-2]:
        binned_thresholds[-2] = right_threshold
        summaries[-2] = summary
        right_index -= 1

    determine_results(
        summaries,
        binned_thresholds,
        percentile_bins,
        left_index,
        right_index,
        split,
        policy,
        envs,
        wandb_logger,
        evaluator,
    )

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
        thresholds=initial_thresholds,
        results=np.array(summaries),
        training_scores=policy.get_train_decision_scores(),
    )


def determine_results(
    summaries: List[dict],
    thresholds: List[float],
    percentile_bins: List[float],
    left_index: int,
    right_index: int,
    split: str,
    policy,
    envs,
    wandb_logger,
    evaluator,
):
    left_threshold = thresholds[left_index]
    right_threshold = thresholds[right_index]

    middle_threshold = left_threshold + (right_threshold - left_threshold) / 2
    summary = evaluator.eval(
        policy,
        envs,
        [split],
        logger=wandb_logger,
        threshold=middle_threshold,
        # percentile_step=percentile_step,
    )
    afhp = summary[split]["action_1_frac"]
    bin_idx = determine_bin(percentile_bins, afhp)
    if summaries[bin_idx] is None:
        summaries[bin_idx] = summary
        thresholds[bin_idx] = middle_threshold

    # Check if empty bins remain left
    left_remaining = bins_remaining(percentile_bins, left_index, bin_idx)
    if left_remaining:
        determine_results(
            summaries,
            thresholds,
            percentile_bins,
            left_index,
            bin_idx,
            split,
            policy,
            envs,
            wandb_logger,
            evaluator,
        )

    right_remaining = bins_remaining(summaries, left_index, bin_idx)
    if right_remaining:
        determine_results(
            summaries,
            thresholds,
            percentile_bins,
            bin_idx,
            right_index,
            split,
            policy,
            envs,
            wandb_logger,
            evaluator,
        )


def determine_bin(percentile_bins, afhp) -> int:
    if afhp < 0.0 or afhp > 1.0:
        raise ValueError(f"Error, encountered an AFHP of {afhp}")
    afhp_percent = afhp * 100
    for i in range(len(percentile_bins) - 1):
        if (
            afhp_percent >= percentile_bins[i]
            and afhp_percent <= percentile_bins[i + 1]
        ):
            return i
    raise ValueError(
        f"Encountered issue with percentile_bins {percentile_bins} "
        f"and afhp {afhp_percent}."
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
        policy.update_params(threshold)
    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )


if __name__ == "__main__":
    main()

from pathlib import Path
import os
import time

from sympy import O
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

    # Determine threshold percentiles
    thresholds, percentile_steps = policy.compute_train_percentiles(
        args.eval.num_thresholds
    )

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

    results = []
    for threshold, percentile_step in zip(thresholds, percentile_steps):
        update_policy_params(policy, threshold)
        summary = evaluator.eval(
            policy,
            envs,
            [split],
            logger=wandb_logger,
            threshold=threshold,
            percentile_step=percentile_step,
        )
        results.append(
            {
                "reward_mean": summary[split]["reward_mean"],
                "reward_std": summary[split]["reward_std"],
                "action_1_frac": summary[split]["action_1_frac"],
                # "threshold": threshold,
                # "percentile_step": percentile_step,
            }
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
        # thresholds=thresholds,
        results=np.array(results),
        training_scores=policy.get_train_decision_scores(),
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


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

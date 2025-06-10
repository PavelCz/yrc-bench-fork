from pathlib import Path
import os
import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

from YRC.policies import *  # noqa: F403
import numpy as np

if __name__ == "__main__":
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        policy.load_model(os.path.join(config.experiment_dir, config.file_name))
    evaluator = Evaluator(config.evaluation)

    # Determine threshold percentiles
    thresholds = policy.compute_train_percentiles(args.eval.num_thresholds)

    # Linearly extend the thresholds below the lowest threshold.
    delta = thresholds[-1] - thresholds[0]
    # Similarly, extend the thresholds above the highest threshold.
    additional_thresholds = []
    highest_threshold = thresholds[-1]
    for i in range(0, args.eval.num_thresholds * 2):
        additional_thresholds.append(highest_threshold + delta * (2**i))
    thresholds = np.concatenate([thresholds, np.array(additional_thresholds)])

    split = "test"

    summaries = []
    for threshold in thresholds:
        params = {"threshold": threshold}
        policy.update_params(params)
        summary = evaluator.eval(policy, envs, [split])
        summaries.append(summary)

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
        thresholds=thresholds,
        results=np.array(summaries),
        training_scores=policy._train_decision_scores,
    )

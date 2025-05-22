import json
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

    # Collect all thresholds on the training set
    score = policy.clf.decision_scores_

    # Determine threshold percentiles
    thresholds = np.percentile(score, np.linspace(0, 100, args.eval.num_thresholds))

    split = "test"

    results = {"thresholds": [], "results": []}
    for threshold in thresholds:
        params = {"threshold": threshold}
        policy.update_params(params)
        summary = evaluator.eval(policy, envs, [split])
        results["thresholds"].append(threshold)
        results["results"].append(summary)

    # Save result summary to file.
    log_file_path = get_global_variable("log_file")
    if log_file_path is None:
        raise ValueError(
            "Log file path is not set. Could not find path to save results."
        )
    log_file_path = Path(log_file_path)
    results_file_path = log_file_path.with_name(
        log_file_path.name.replace(".log", f"_{split}.json")
    )
    with results_file_path.open("w") as f:
        json.dump(results, f, indent=2)

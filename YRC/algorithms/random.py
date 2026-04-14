import numpy as np
import logging
from YRC.core import Algorithm
from YRC.core.configs.global_configs import get_global_variable
from YRC.policies.base import OracleLevelBasedRandomPolicy
from typing import List, Optional
import torch


class RandomAlgorithm(Algorithm):
    def __init__(self, config, env):
        self.args = config

    def train(
        self,
        policy,
        envs,
        rollout_obs: Optional[List[torch.Tensor]],
        evaluator=None,
        train_split=None,
        eval_splits=None,
    ):
        save_dir = get_global_variable("experiment_dir")

        best_summary = {}
        for split in eval_splits:
            best_summary[split] = {"reward_mean": -1e9}

        best_prob = {}
        if isinstance(policy, OracleLevelBasedRandomPolicy):
            cand_probs = list(np.arange(0.0, 2.1, 0.1))
        else:
            cand_probs = list(np.arange(0.0, 1.1, 0.1))

        logging.info("Candidate probs: " + str(cand_probs))

        for prob in cand_probs:
            logging.info(f"Prob: {prob}")

            policy.update_params(prob)
            split_summary = evaluator.eval(policy, envs, eval_splits)

            for split in eval_splits:
                if (
                    split_summary[split]["reward_mean"]
                    > best_summary[split]["reward_mean"]
                ):
                    best_prob[split] = prob
                    best_summary[split] = split_summary[split]
                    policy.save_model(f"best_{split}", save_dir)

                # log best result so far
                logging.info(f"Best {split} so far")
                logging.info(f"Prob: {best_prob[split]}")
                evaluator.write_summary(f"best_{split}", best_summary[split])

        policy.update_params(best_prob)

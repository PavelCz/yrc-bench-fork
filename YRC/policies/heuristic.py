import logging
import numpy as np

import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.optim as optim

import os

from YRC.core.policy import Policy
import YRC.models as models
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.configs.utils import config_logging



class ExponentialHeuristicPolicy(Policy):
    def __init__(self, config, env):
        self.non_ood_starting_prob = 0.5
        self.device = get_global_variable("device")
        self.timestep = 0
        
    def reset_episode(self):
        """Reset the timestep counter at the start of a new episode."""
        self.timestep = 0

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        benchmark = get_global_variable("benchmark")
        env_obs = obs["env_obs"]

        if isinstance(env_obs, dict):
            if benchmark == "cliport":
                action_shape = (1,)
            elif benchmark == "minigrid":
                action_shape = (env_obs["direction"].shape[0],)
        else:
            action_shape = (env_obs.shape[0],)

        # Calculate probability: P(detect OOD) = 1 - non_ood_starting_prob^t
        # Therefore: P(not detect OOD) = non_ood_starting_prob^t
        current_non_ood_prob = self.non_ood_starting_prob ** self.timestep
        current_prob = 1 - current_non_ood_prob
        
        # Increment timestep for next call
        self.timestep += 1

        action = torch.rand(action_shape).to(self.device) < current_prob
        action = action.int()

        if return_scores_and_recons:
            return action.cpu().numpy(), None, None

        return action.cpu().numpy()

    def update_params(self, prob=None):
        if prob is None:
            raise ValueError("Probability cannot be None!")
        ood_starting_prob = prob

        self.non_ood_starting_prob = 1 - ood_starting_prob

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save({"prob": self.non_ood_starting_prob}, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.non_ood_starting_prob = ckpt["prob"]

    def train_percentile(self, percentile: float) -> float:
        """Take a percentile and return the threshold for that percentile."""
        return (100 - percentile) * 0.01
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


class WaitPolicy(Policy):
    """
    Simple heuristic: wait for n timesteps, then always ask for help.

    The threshold parameter controls n (number of timesteps to wait).
    """

    def __init__(self, config, env):
        self.device = get_global_variable("device")
        self.timestep = 0
        self.threshold = 0  # Number of timesteps to wait before asking

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

        # Ask for help if we've waited enough timesteps
        should_ask = self.timestep >= self.threshold

        # Increment timestep for next call
        self.timestep += 1

        if should_ask:
            action = torch.ones(action_shape, dtype=torch.int, device=self.device)
        else:
            action = torch.zeros(action_shape, dtype=torch.int, device=self.device)

        if return_scores_and_recons:
            return action.cpu().numpy(), None, None

        return action.cpu().numpy()

    def update_params(self, threshold=None):
        if threshold is None:
            raise ValueError("Threshold cannot be None!")
        self.threshold = int(threshold)

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save({"threshold": self.threshold}, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.threshold = ckpt["threshold"]

    def train_percentile(self, percentile: float) -> float:
        """
        Take a percentile and return the threshold for that percentile.

        For this policy, we map percentile to timesteps.
        Higher percentile = ask for help more = lower threshold (wait less).
        At 0% AFHP, threshold is very high (never ask).
        At 100% AFHP, threshold is 0 (always ask from start).

        We use a max episode length of 1000 as reference.
        """
        max_timesteps = 1000
        # Invert: 0% -> max_timesteps, 100% -> 0
        threshold = int(max_timesteps * (100 - percentile) / 100)
        return threshold
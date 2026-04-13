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
        self._mean_episode_length = None
        
    def reset_episode(self, env_idx: int = None):
        """Reset the timestep counter at the start of a new episode.

        Args:
            env_idx: Ignored for this policy (uses single counter for simplicity
                    since the policy is stochastic anyway).
        """
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

    def train_percentile_step(self, percentile: float) -> float:
        raise NotImplementedError(
            "ExponentialHeuristicPolicy does not support step_afhp calibration."
        )

    def train_percentile_level(self, percentile: float) -> float:
        """Map percentile to ood_starting_prob calibrated for level_afhp.

        At timestep t, P(no help) = (1 - ood_starting_prob)^t.
        Over an episode of length L:
            P(no help in episode) = product_{t=0}^{L-1} (1 - ood_starting_prob)^t
                                  = (1 - ood_starting_prob)^{L(L-1)/2}

        Inverting: ood_starting_prob = 1 - (percentile/100)^{2/(L(L-1))}

        Falls back to linear mapping if mean episode length is not calibrated.
        """
        if self._mean_episode_length is not None:
            p = percentile / 100.0
            p = max(0.0, min(1.0, p))
            if p <= 0.0:
                return 1.0  # always ask
            if p >= 1.0:
                return 0.0  # never ask
            L = self._mean_episode_length
            exponent = 2.0 / (L * (L - 1))
            return 1.0 - p**exponent
        return (100 - percentile) * 0.01


class WaitPolicy(Policy):
    """
    Simple heuristic: wait for n timesteps, then always ask for help.

    The threshold parameter controls n (number of timesteps to wait).
    Maintains per-environment timestep counters for vectorized environments.
    """

    def __init__(self, config, env):
        self.device = get_global_variable("device")
        self.num_envs = env.num_envs
        self.timesteps = np.zeros(self.num_envs, dtype=np.int32)
        self.threshold = 0  # Number of timesteps to wait before asking
        self._episode_lengths = None

        # Get max episode length from config for threshold sampling
        max_steps = getattr(config.environment.common, "max_steps", None)
        if max_steps is None:
            # Default Procgen timeout is 1000, but many envs use 500
            max_steps = 1000
        self.max_episode_length = max_steps

    def reset_episode(self, env_idx: int = None):
        """Reset the timestep counter at the start of a new episode.

        Args:
            env_idx: If provided, reset only that environment's counter.
                    If None, reset all counters (for compatibility).
        """
        if env_idx is not None:
            self.timesteps[env_idx] = 0
        else:
            self.timesteps[:] = 0

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        benchmark = get_global_variable("benchmark")
        env_obs = obs["env_obs"]

        if isinstance(env_obs, dict):
            if benchmark == "cliport":
                num_envs = 1
            elif benchmark == "minigrid":
                num_envs = env_obs["direction"].shape[0]
        else:
            num_envs = env_obs.shape[0]

        # Ask for help if we've waited enough timesteps (per environment)
        should_ask = self.timesteps[:num_envs] >= self.threshold

        # Increment timesteps for next call
        self.timesteps[:num_envs] += 1

        action = torch.tensor(should_ask.astype(np.int32), device=self.device)

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

    def train_percentile_step(self, percentile: float) -> float:
        """Map percentile to a wait-timestep threshold for step_afhp.

        Note: percentile_to_threshold already inverts the target AFHP before
        calling this. So if we want 10% AFHP, this receives percentile=90.

        To achieve X% AFHP: threshold = episode_length * (100-X) / 100
        Since we receive (100-X) as percentile, we just use percentile directly.

        Uses max_episode_length from environment config.
        """
        # percentile is already inverted (90 means want 10% AFHP)
        # threshold = episode_length * percentile / 100
        threshold = int(self.max_episode_length * percentile / 100)
        return threshold

    def train_percentile_level(self, percentile: float) -> float:
        """Map percentile to a wait-timestep threshold for level_afhp.

        An episode has help iff its length > threshold. So the threshold
        at the p-th percentile of episode lengths is where (100-p)% of
        episodes are long enough to receive help, i.e., level_afhp = (100-p)%.

        Requires calibration: _episode_lengths must be set from training data.
        """
        if self._episode_lengths is None:
            raise ValueError(
                "Episode lengths not calibrated. "
                "Run python -m apps.calibrate_afhp first."
            )
        return np.percentile(self._episode_lengths, percentile)
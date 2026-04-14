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


class BasePolicy(Policy):
    def __init__(self, config, env):
        self.model_cls = getattr(models, config.coord_policy.model_cls)
        self.model = self.model_cls(config, env)
        self.model.to(get_global_variable("device"))
        self.optim = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-5)
        # TODO: not sure write optim.Adam messes up logging, need to reconfigure here
        config_logging(get_global_variable("log_file"))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def predict(self, obs):
        logit = self.model(obs)
        log_prob = F.log_softmax(logit, dim=-1)
        return Categorical(logits=log_prob)

    def act(self, obs, greedy=False):
        dist = self.predict(obs)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.cpu().numpy()

    def update_params(self, grad_clip_norm=None):
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad()

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
            },
            save_path,
        )
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optim_state_dict"])


class AlwaysPolicy(Policy):
    def __init__(self, config, env):
        agent = config.coord_policy.agent
        assert agent in ["weak", "strong"], f"Unrecognized agent: {agent}!"
        self.choice = env.WEAK if agent == "weak" else env.STRONG

    def act(self, obs, greedy=False):
        benchmark = get_global_variable("benchmark")
        env_obs = obs["env_obs"]

        if isinstance(env_obs, dict):
            if benchmark == "cliport":
                action_shape = (1,)
            elif benchmark == "minigrid":
                action_shape = (env_obs["direction"].shape[0],)
        else:
            action_shape = (env_obs.shape[0],)

        action = np.ones(action_shape, dtype=np.int64) * self.choice
        return action


class TimestepRandomPolicy(Policy):
    def __init__(self, config, env):
        self.prob = 0.5
        self.device = get_global_variable("device")
        self._mean_episode_length = None

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

        action = torch.rand(action_shape).to(self.device) < self.prob
        action = action.int()

        if return_scores_and_recons:
            return action.cpu().numpy(), None, None

        return action.cpu().numpy()

    def update_params(self, prob=None):
        if prob is None:
            raise ValueError("Probability cannot be None!")
        self.prob = prob

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save({"prob": self.prob}, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.prob = ckpt["prob"]

    def train_percentile_step(self, percentile: float) -> float:
        """Map percentile to per-step ask-for-help probability (linear)."""
        return (100 - percentile) * 0.01

    def train_percentile_level(self, percentile: float) -> float:
        """Map percentile to ask-for-help probability calibrated for level_afhp.

        If mean episode length has been calibrated, uses the formula:
            prob = 1 - (percentile / 100) ^ (1 / L)
        to account for the nonlinear mapping between per-step probability
        and per-episode OOD percentage.

        Otherwise falls back to linear mapping.
        """
        if self._mean_episode_length is not None:
            p = percentile / 100.0
            # Clamp to avoid domain errors
            p = max(0.0, min(1.0, p))
            if p <= 0.0:
                return 1.0  # always ask
            if p >= 1.0:
                return 0.0  # never ask
            return 1.0 - p ** (1.0 / self._mean_episode_length)
        return (100 - percentile) * 0.01


class LevelBasedRandomPolicy(Policy):
    """A random policy that checks once at the beginning of the episode, whether it
    should ask for help or not. It then sticks with that decision for the rest of the
    episode.
    """

    def __init__(self, config, env):
        self.prob = 0.5
        self.device = get_global_variable("device")

        self.current_action = [None] * env.num_envs

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        for i, ep_timestep in enumerate(obs["episode_timestep"]):
            if ep_timestep == 0:
                # Randomly sample a new action at the beginning of the episode.
                action = torch.rand(1).item() < self.prob
                action = int(action)
                self.current_action[i] = action

        if return_scores_and_recons:
            return np.array(self.current_action), None, None

        return np.array(self.current_action)

    def update_params(self, prob=None):
        if prob is None:
            raise ValueError("Probability cannot be None!")
        self.prob = prob

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save({"prob": self.prob}, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.prob = ckpt["prob"]

    def train_percentile_step(self, percentile: float) -> float:
        raise NotImplementedError(
            "LevelBasedRandomPolicy does not support step_afhp calibration."
        )

    def train_percentile_level(self, percentile: float) -> float:
        """Take a percentile and return the threshold for that percentile."""
        return (100 - percentile) * 0.01


class OracleLevelBasedRandomPolicy(Policy):
    """A delayed oracle-gated level policy.

    On the first action of an episode, the policy always stays with the weak agent.
    Once the first info dict has exposed the level's OOD ground truth, the policy
    latches a per-episode help decision:

    - control in [0, 1): ask on OOD levels with probability ``control``
    - control in [1, 2]: always ask on OOD levels and ask on ID levels with
      probability ``control - 1``

    The latched choice is then reused for the rest of the episode.
    """

    def __init__(self, config, env):
        self.help_control = 1.0
        self.current_action = [0] * env.num_envs

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        episode_timesteps = obs["episode_timestep"]
        level_ood_gt = obs["level_ood_gt"]

        for i, ep_timestep in enumerate(episode_timesteps):
            if ep_timestep == 0:
                # We only know the OOD label after the first env step.
                self.current_action[i] = 0
                continue

            if ep_timestep == 1:
                is_ood_level = bool(level_ood_gt[i])
                if self.help_control < 1.0:
                    ask_for_help = is_ood_level and (
                        torch.rand(1).item() < self.help_control
                    )
                else:
                    ask_for_help = is_ood_level or (
                        torch.rand(1).item() < (self.help_control - 1.0)
                    )
                self.current_action[i] = int(ask_for_help)

        if return_scores_and_recons:
            return np.array(self.current_action), None, None

        return np.array(self.current_action)

    def update_params(self, help_control=None):
        if help_control is None:
            raise ValueError("Help control cannot be None!")
        self.help_control = float(np.clip(help_control, 0.0, 2.0))

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save({"help_control": self.help_control}, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        if "help_control" in ckpt:
            self.help_control = ckpt["help_control"]
        elif "prob" in ckpt:
            # Backward-compatible fallback for early experiments.
            self.help_control = ckpt["prob"]
        else:
            raise KeyError("Checkpoint missing 'help_control' entry")

    def train_percentile_step(self, percentile: float) -> float:
        raise NotImplementedError(
            "OracleLevelBasedRandomPolicy does not support step_afhp calibration."
        )

    def train_percentile_level(self, percentile: float) -> float:
        """Map percentile directly to the [0, 2] help-control range.

        This assumes a 50/50 mix of OOD and ID evaluation levels, so the control maps
        linearly to level AFHP over the full [0, 100] range.
        """
        return (100 - percentile) * 0.02

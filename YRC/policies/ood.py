import os
import numpy as np
from copy import deepcopy as dc
import logging
import collections
from YRC.core import Policy
from lib.pyod.pyod.models import deep_svdd
from joblib import dump, load
from YRC.core.configs.global_configs import get_global_variable
from YRC.models.utils import AutoEncoderWithVal
from YRC.core.utils import to_tensor
from typing import Optional


class OODPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        if config.coord_policy.collect_data_agent == "weak":
            self.agent = env.weak_agent
        elif config.coord_policy.collect_data_agent == "strong":
            self.agent = env.strong_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0}
        self.clf = None
        self.clf_name = None
        self.device = get_global_variable("device")
        self.feature_type = config.coord_policy.feature_type
        self.logger = None  # Will be set by train_svdd.py for wandb logging

        # Store training scores for percentile computation (used in AFHP eval)
        self._train_scores = None
        self._train_episode_max_scores = None

        # Rolling average setup
        self.rolling_average: Optional[str] = getattr(
            self.args, "rolling_average", None
        )
        if self.rolling_average == "none":
            self.rolling_average = None

        if (
            self.rolling_average is not None
            and self.rolling_average != "mean"
            and self.rolling_average != "median"
        ):
            raise ValueError(f"Rolling average {self.rolling_average} not supported")

        self.rolling_average_size: int = getattr(self.args, "rolling_average_size", 10)
        self.rolling_average_buffers = []

        if self.rolling_average is not None:
            for _ in range(env.num_envs):
                self.rolling_average_buffers.append(
                    collections.deque(
                        self.rolling_average_size * [float("-inf")],
                        self.rolling_average_size,
                    )
                )

    def update_params(self, params):
        self.params = dc(params)
        if "threshold" not in params:
            raise ValueError(
                "Threshold is not in the provided params. "
                "You're probably doing something wrong"
            )
        self.clf.threshold_ = params["threshold"]

    def fit(self, x, x_threshold, y=None, x_val=None, batch_transform=None):
        if self.clf_name == "DeepSVDD":
            # We don't want to move the complete datasets to the GPU, we move the
            # the batches separately.
            # x = x.to(self.device)
            # x_threshold = x_threshold.to(self.device)
            self.clf.fit(
                X=x,
                X_threshold=x_threshold,
                y=y,
                X_val=x_val,
                batch_transform=batch_transform,
            )
        elif self.clf_name == "AutoEncoder":
            x = x.cpu()
            # Flatten the observations.
            x = x.reshape(x.shape[0], -1)

            val_data = x_val if x_val is not None else x_threshold
            val_data = val_data.cpu()
            val_data = val_data.reshape(val_data.shape[0], -1)

            self.clf.set_loaders(x, val_data)
            self.clf.fit(x, y)
        else:
            raise ValueError(f"Unknown OOD detector type: {self.clf_name}")

    def _prepare_observation(self, obs):
        """Extract and format observation for the OOD detector.

        Handles feature type selection, benchmark-specific preprocessing, and
        backend-specific formatting (AutoEncoder needs CPU+flat, DeepSVDD needs GPU).
        """
        keys = {
            "obs": ["env_obs"],
            "hidden": ["weak_features"],
            "dist": ["weak_logit"],
            "hidden_obs": ["env_obs", "weak_features"],
            "hidden_dist": ["weak_features", "weak_logit"],
            "obs_dist": ["env_obs", "weak_logit"],
            "obs_hidden_dist": ["env_obs", "weak_features", "weak_logit"],
        }[self.feature_type]

        if get_global_variable("benchmark") in ["cliport", "minigrid"]:
            observation = [
                to_tensor(
                    obs[key]["image"] if key == "env_obs" else to_tensor(obs[key])
                )
                for key in keys
            ]
        else:
            observation = [to_tensor(obs[key]) for key in keys]

        if self.feature_type in ["obs", "hidden", "dist"]:
            observation = observation[0]

        if self.clf_name == "AutoEncoder":
            observation = observation.cpu()
            observation = observation.reshape(observation.shape[0], -1)
        elif self.clf_name == "DeepSVDD":
            observation = observation.to(self.device)

        return observation

    def _compute_scores(self, obs):
        """Compute OOD decision scores for the given observation dict."""
        observation = self._prepare_observation(obs)
        return self.clf.decision_function(observation)

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        score = self._compute_scores(obs)

        # Store original scores before applying rolling average
        score_original = score.copy() if self.rolling_average is not None else None

        # Apply rolling average if enabled
        if self.rolling_average is not None:
            for i in range(len(self.rolling_average_buffers)):
                self.rolling_average_buffers[i].append(score[i])

                if self.rolling_average == "mean":
                    score[i] = np.mean(self.rolling_average_buffers[i])
                elif self.rolling_average == "median":
                    score[i] = np.median(self.rolling_average_buffers[i])
                else:
                    raise NotImplementedError(
                        f"Unrecognized rolling average: {self.rolling_average}"
                    )

        # Store the scores for potential retrieval (used by evaluator for histograms)
        self.last_scores_original = score_original
        self.last_scores_rolling_avg = (
            score if self.rolling_average is not None else None
        )

        action = 1 - (score < self.clf.threshold_).astype(int)
        if 0 not in action and 1 not in action:
            print("No action is selected as OOD")

        if return_scores_and_recons:
            return action, score, None

        return action

    def generate_scores(self, env, num_rollouts):
        """Run rollouts to collect per-step and per-episode-max OOD scores.

        Similar to ThresholdPolicy.generate_scores(), but uses the OOD detector's
        decision_function instead of confidence scores. The weak agent is used to
        select actions (via its logits) while we record the OOD scores at each step.
        """
        from torch.distributions.categorical import Categorical

        scores = []
        episode_max_scores = []
        current_episode_max_scores = [float("-inf")] * env.num_envs
        seeds_exhausted = np.array([False] * env.num_envs)
        progress_interval = max(env.num_envs, env.num_envs * 10)
        next_progress = min(progress_interval, num_rollouts)
        vector_steps = 0
        max_vector_steps = max(
            10000, ((num_rollouts + env.num_envs - 1) // env.num_envs) * 1000
        )

        agent = self.agent
        agent.eval()

        if hasattr(env, "get_obs") and getattr(env, "env_obs", None) is not None:
            obs = env.get_obs()
        else:
            obs = env.reset()

        logging.info(
            "Generating OOD calibration scores for "
            f"{num_rollouts} rollouts (num_envs={env.num_envs})"
        )

        while len(episode_max_scores) < num_rollouts:
            score = self._compute_scores(obs)

            for i in range(env.num_envs):
                if seeds_exhausted[i]:
                    continue
                score_i = score.item() if env.num_envs == 1 else score[i].item()
                scores.append(score_i)
                current_episode_max_scores[i] = max(
                    current_episode_max_scores[i], score_i
                )

            logit = agent.forward(obs["env_obs"])
            dist = Categorical(logits=logit / self.params["explore_temp"])
            action = dist.sample().cpu().numpy()

            obs, reward, done, info = env.step(action)
            vector_steps += 1

            for i in range(env.num_envs):
                if done[i] and not seeds_exhausted[i]:
                    episode_max_scores.append(current_episode_max_scores[i])
                    current_episode_max_scores[i] = float("-inf")
                    if len(episode_max_scores) >= num_rollouts:
                        break

            for i in range(env.num_envs):
                if info[i].get("seeds_exhausted", False):
                    seeds_exhausted[i] = True

            if len(episode_max_scores) >= next_progress:
                logging.info(
                    "OOD calibration score progress: "
                    f"{len(episode_max_scores)}/{num_rollouts} episodes, "
                    f"{len(scores)} scores, {vector_steps} vector steps"
                )
                while next_progress <= len(episode_max_scores):
                    next_progress += progress_interval

            if seeds_exhausted.all():
                logging.info(
                    "All calibration environments exhausted their sequential seeds. "
                    f"Collected {len(episode_max_scores)}/{num_rollouts} episodes."
                )
                break

            if vector_steps % 10000 == 0:
                logging.warning(
                    "OOD calibration still running after "
                    f"{vector_steps} vector steps; collected "
                    f"{len(episode_max_scores)}/{num_rollouts} episodes"
                )

            if vector_steps >= max_vector_steps:
                raise RuntimeError(
                    "OOD calibration exceeded "
                    f"{max_vector_steps} vector steps while collecting "
                    f"{len(episode_max_scores)}/{num_rollouts} episodes. "
                    "This usually means at least one vectorized environment did not "
                    "emit done or seeds_exhausted during calibration."
                )

        if len(episode_max_scores) < num_rollouts:
            logging.warning(
                "OOD calibration stopped before collecting the requested number of "
                f"episodes: {len(episode_max_scores)}/{num_rollouts}. "
                "This can happen when the calibration environment was already reset "
                "before score generation in sequential seed mode."
            )

        self._train_scores = np.array(scores)
        self._train_episode_max_scores = np.array(episode_max_scores)
        logging.info(
            "Generated OOD calibration scores: "
            f"{len(scores)} step scores, {len(episode_max_scores)} episode max scores"
        )
        return scores

    def _rollout_once(self, env, batch_idx=None):
        from torch.distributions.categorical import Categorical

        agent = self.agent
        agent.eval()

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        scores = []
        episode_max_scores = [float("-inf")] * env.num_envs
        steps = 0
        max_steps = 10000
        batch_label = "" if batch_idx is None else f" in batch {batch_idx + 1}"

        while not has_done.all():
            score = self._compute_scores(obs)

            if env.num_envs == 1:
                scores.append(score.item())
                episode_max_scores[0] = max(episode_max_scores[0], score.item())
            else:
                for i in range(env.num_envs):
                    if not has_done[i]:
                        scores.append(score[i].item())
                        episode_max_scores[i] = max(
                            episode_max_scores[i], score[i].item()
                        )

            # Use weak agent to select actions for stepping the environment
            logit = agent.forward(obs["env_obs"])
            dist = Categorical(logits=logit / self.params["explore_temp"])
            action = dist.sample().cpu().numpy()

            obs, reward, done, info = env.step(action)
            steps += 1
            has_done |= done

            if steps % 1000 == 0:
                unfinished = np.flatnonzero(~has_done).tolist()
                logging.warning(
                    "OOD calibration rollout still running after "
                    f"{steps} steps"
                    f"{batch_label}; "
                    f"unfinished env indices: {unfinished}"
                )

            if steps >= max_steps and not has_done.all():
                unfinished = np.flatnonzero(~has_done).tolist()
                raise RuntimeError(
                    "OOD calibration rollout exceeded "
                    f"{max_steps} steps"
                    f"{batch_label}; "
                    f"unfinished env indices: {unfinished}. "
                    "This usually means at least one vectorized environment did not "
                    "emit done during calibration."
                )

        return scores, episode_max_scores

    def initialize_ood_detector(self, args, env):
        dummy_obs = env.reset()
        feature_type_to_shapes = {
            "obs": lambda dummy_obs: (
                dummy_obs["env_obs"]["image"]
                if get_global_variable("benchmark") in ["cliport", "minigrid"]
                else dummy_obs["env_obs"]
            ).shape,
            "hidden": lambda dummy_obs: dummy_obs["weak_features"].shape,
            "hidden_obs": lambda dummy_obs: (
                (
                    dummy_obs["env_obs"]["image"]
                    if get_global_variable("benchmark") in ["cliport", "minigrid"]
                    else dummy_obs["env_obs"]
                ).shape
                + dummy_obs["weak_features"].shape[1:]
            ),
            "dist": lambda dummy_obs: dummy_obs["weak_logit"].shape,
            "hidden_dist": lambda dummy_obs: (
                dummy_obs["weak_features"].shape + dummy_obs["weak_logit"].shape[1:]
            ),
            "obs_dist": lambda dummy_obs: (
                (
                    dummy_obs["env_obs"]["image"]
                    if get_global_variable("benchmark") in ["cliport", "minigrid"]
                    else dummy_obs["env_obs"]
                ).shape
                + dummy_obs["weak_logit"].shape[1:]
            ),
            "obs_hidden_dist": lambda dummy_obs: (
                (
                    dummy_obs["env_obs"]["image"]
                    if get_global_variable("benchmark") in ["cliport", "minigrid"]
                    else dummy_obs["env_obs"]
                ).shape
                + dummy_obs["weak_features"].shape[1:]
                + dummy_obs["weak_logit"].shape[1:]
            ),
        }

        dummy_obs_shape = feature_type_to_shapes[self.feature_type](dummy_obs)

        if self.args.method == "DeepSVDD":
            self.clf_name = "DeepSVDD"
            self.clf = deep_svdd.DeepSVDD(
                n_features=args.feature_size,
                use_ae=args.use_ae,
                contamination=args.contamination,
                epochs=args.epoch,
                batch_size=args.batch_size,
                input_shape=dummy_obs_shape,
                feature_type=self.feature_type,
                benchmark=get_global_variable("benchmark"),
                logger=self.logger,
            )
            self.clf.model_.to(self.device)
        elif self.args.method == "AutoEncoder":
            self.clf_name = "AutoEncoder"
            clf = AutoEncoderWithVal(
                contamination=args.contamination,
                epoch_num=args.epoch,
                batch_size=args.batch_size,
                device=self.device,
                preprocessing=False,
            )
            self.clf = clf
        else:
            raise ValueError(f"Unknown OOD detector type: {args.ood_detector}")

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.joblib")
        state_dict = {
            "clf": self.clf,
            "class_name": self.__class__.__name__,
            "config": {
                "contamination": self.clf.contamination,
            },
            "clf_name": self.clf_name,
        }
        if isinstance(self.clf, deep_svdd.DeepSVDD):
            state_dict["config"]["use_ae"] = self.clf.use_ae
        dump(state_dict, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_dir):
        state_dict = load(f"{load_dir}")
        self.clf = state_dict["clf"]
        self.clf_name = state_dict["clf_name"]
        logging.info(f"Loaded OOD model from {load_dir}")

        return self

    def reset_rolling_average_buffer(self, index: int) -> None:
        """Reset the rolling average buffer for a given index. The index corresponds
        to the environment index.
        """
        if self.rolling_average is not None:
            self.rolling_average_buffers[index] = collections.deque(
                self.rolling_average_size * [float("-inf")], self.rolling_average_size
            )

    def train_percentile_step(self, percentile: float) -> float:
        """Return threshold for a target step_afhp percentile.

        Uses rollout-based per-step scores if available (from generate_scores()),
        otherwise falls back to decision scores from OOD detector training.
        """
        if self._train_scores is not None:
            return np.percentile(self._train_scores, percentile)
        return np.percentile(self.clf.decision_scores_, percentile)

    def train_percentile_level(self, percentile: float) -> float:
        """Return threshold for a target level_afhp percentile.

        Uses per-episode max scores from generate_scores() rollouts.
        """
        if self._train_episode_max_scores is None:
            raise ValueError(
                "Episode-level training scores not available. "
                "Call generate_scores() first."
            )
        return np.percentile(self._train_episode_max_scores, percentile)

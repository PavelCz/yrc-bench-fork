import collections
import importlib
import logging
import os
from copy import deepcopy as dc
from typing import Optional

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from YRC.core import Policy
from YRC.core.configs.global_configs import get_global_variable


class ThresholdPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        self.agent = env.weak_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0, "score_temp": 1.0}
        self.device = get_global_variable("device")
        self.rolling_average: Optional[str] = self.args.rolling_average

        if self.rolling_average == "none":
            self.rolling_average = None

        if (
            self.rolling_average is not None
            and self.rolling_average != "mean"
            and self.rolling_average != "median"
        ):
            raise ValueError(f"Rolling average {self.rolling_average} not supported")

        self.rolling_average_size: int = self.args.rolling_average_size

        self.rolling_average_buffers = []

        if self.rolling_average is not None:
            for _ in range(env.num_envs):
                self.rolling_average_buffers.append(
                    collections.deque(
                        self.rolling_average_size * [float("-inf")],
                        self.rolling_average_size,
                    )
                )

        # Store training scores for percentile computation (used in AFHP eval)
        self._train_scores = None
        self._train_episode_max_scores = None

        # Ensemble for ensemble_variance metric
        self.ensemble_members = None
        self._single_weak_agent = None  # Original weak agent (for ensemble_use_single_weak)
        if self.args.metric == "ensemble_variance":
            ensemble_paths = getattr(self.args, "ensemble_members", None)
            if not ensemble_paths:
                raise ValueError(
                    "ensemble_variance metric requires -cp_ensemble_members"
                )

            # Load ensemble members
            benchmark = get_global_variable("benchmark")
            module = importlib.import_module(f"YRC.envs.{benchmark}")
            load_fn = getattr(module, "load_policy")

            # Store original weak agent for optional use in action selection
            self._single_weak_agent = self.agent

            # Start with weak agent as first ensemble member
            members = [self.agent]
            for path in ensemble_paths:
                member = load_fn(path, env.base_env)
                member.eval()
                members.append(member)

            # Create EnsemblePolicy wrapper and replace self.agent
            EnsemblePolicy = getattr(module, "EnsemblePolicy")
            self.agent = EnsemblePolicy(members)
            self.ensemble_members = members

            use_single = getattr(self.args, "ensemble_use_single_weak", False)
            logging.info(
                f"Loaded ensemble with {len(members)} members "
                f"(weak agent + {len(ensemble_paths)} additional), "
                f"use_single_weak={use_single}"
            )

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        if get_global_variable("benchmark") == "cliport":
            attention_size = 3  # todo: get this shape automatically
            attention_flat = obs["weak_logit"][:, :attention_size]
            transport_flat = obs["weak_logit"][:, attention_size:]
            if not torch.is_tensor(attention_flat):
                attention_flat = (
                    torch.from_numpy(attention_flat).float().to(self.device)
                )
            if not torch.is_tensor(transport_flat):
                transport_flat = (
                    torch.from_numpy(transport_flat).float().to(self.device)
                )
            attention_score = self._compute_score(attention_flat)
            transport_score = self._compute_score(transport_flat)
            score = torch.mean(
                torch.stack([attention_score, transport_score])
            ).unsqueeze(0)
        elif self.args.metric == "ensemble_variance":
            env_obs = obs["env_obs"]
            score = self._compute_ensemble_score(env_obs)
        else:
            weak_logit = obs["weak_logit"]
            if not torch.is_tensor(weak_logit):
                weak_logit = torch.from_numpy(weak_logit).float().to(self.device)
            score = self._compute_score(weak_logit)
        # NOTE: Originally, higher score = more certain
        # I inverted score, so it is in line with other OOD scores.
        action = (score > self.params["threshold"]).int()

        if return_scores_and_recons:
            return action.cpu().numpy(), score.detach().cpu().numpy(), None

        return action.cpu().numpy()

    def generate_scores(self, env, num_rollouts):
        assert num_rollouts % env.num_envs == 0
        scores = []
        episode_max_scores = []
        for i in range(num_rollouts // env.num_envs):
            ep_scores, ep_maxes = self._rollout_once(env)
            scores.extend(ep_scores)
            episode_max_scores.extend(ep_maxes)
        # Store scores for percentile computation (used in AFHP eval)
        self._train_scores = np.array(scores)
        self._train_episode_max_scores = np.array(episode_max_scores)
        return scores

    def _rollout_once(self, env):
        def sample_action(logit):
            dist = Categorical(logits=logit / self.params["explore_temp"])
            return dist.sample().cpu().numpy()

        agent = self.agent
        agent.eval()

        # Determine which agent to use for actions
        use_single_weak = getattr(self.args, "ensemble_use_single_weak", False)
        if use_single_weak and self._single_weak_agent is not None:
            action_agent = self._single_weak_agent
            action_agent.eval()
        else:
            action_agent = agent

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        scores = []
        episode_max_scores = [float("-inf")] * env.num_envs

        while not has_done.all():
            if self.args.metric == "ensemble_variance":
                score = self._compute_ensemble_score(obs["env_obs"])
                logit = action_agent.forward(obs["env_obs"])
            else:
                logit = agent.forward(obs["env_obs"])
                score = self._compute_score(logit)

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

            action = sample_action(logit)
            obs, reward, done, info = env.step(action)
            has_done |= done

        return scores, episode_max_scores

    def _compute_score(self, logit):
        # NOTE: higher score = more certain for some of the metrics, but not all.
        metric = self.args.metric
        logit = logit / self.params["score_temp"]
        if metric == "max_logit":
            score = logit.max(dim=-1)[0]
            # Invert the score such that higher score = more ood.
            score = -score
        elif metric == "max_prob":
            # Softmax probability -> how certain are we about that action?
            # Max returns values, indices, which is why we index into the values.
            score = logit.softmax(dim=-1).max(dim=-1)[0]
            # Invert the score such that higher score = more ood.
            score = 1.0 - score
        elif metric == "margin":
            raise NotImplementedError(
                f"Margin metric not implemented for threshold policy"
            )
            if logit.size(-1) > 1:
                # Original behavior for multi-class case
                top2 = logit.softmax(dim=-1).topk(2, dim=-1)[0]
                if len(top2.shape) == 1:
                    top2 = top2.unsqueeze(0)
                score = top2[:, 0] - top2[:, 1]
            else:
                # Binary case when logit has shape (..., 1)
                score = logit.sigmoid().squeeze(-1)
        elif metric == "neg_entropy":
            raise NotImplementedError(
                f"Neg entropy metric not implemented for threshold policy"
            )
            score = -Categorical(logits=logit).entropy()
        elif metric == "neg_energy":
            raise NotImplementedError(
                f"Neg energy metric not implemented for threshold policy"
            )
            score = logit.logsumexp(dim=-1)
        elif metric == "ensemble_variance":
            raise ValueError("ensemble_variance should use _compute_ensemble_score")
        else:
            raise NotImplementedError(f"Unrecognized metric: {metric}")

        # Store original scores before applying rolling average
        score_original = score.clone() if self.rolling_average is not None else None

        if self.rolling_average is not None:
            for i in range(len(self.rolling_average_buffers)):
                self.rolling_average_buffers[i].append(score[i].item())

                if self.rolling_average == "mean":
                    score[i] = torch.mean(
                        torch.tensor(self.rolling_average_buffers[i])
                    ).item()
                elif self.rolling_average == "median":
                    score[i] = torch.median(
                        torch.tensor(self.rolling_average_buffers[i])
                    ).item()
                else:
                    raise NotImplementedError(
                        f"Unrecognized rolling average: {self.rolling_average}"
                    )

        # Store the scores for potential retrieval (used by evaluator for histograms)
        self.last_scores_original = score_original
        self.last_scores_rolling_avg = (
            score if self.rolling_average is not None else None
        )

        return score

    def _compute_ensemble_score(self, obs):
        """Compute variance of softmax outputs across ensemble members."""
        with torch.no_grad():
            member_logits = [m.forward(obs) for m in self.ensemble_members]
            stacked = torch.stack(member_logits)  # [M, B, A]
            probs = torch.softmax(stacked / self.params["score_temp"], dim=-1)
            variance = torch.var(probs, dim=0)  # [B, A]
            score = variance.mean(dim=-1)  # [B]

        # Store original scores before applying rolling average
        score_original = score.clone() if self.rolling_average is not None else None

        if self.rolling_average is not None:
            for i in range(len(self.rolling_average_buffers)):
                self.rolling_average_buffers[i].append(score[i].item())

                if self.rolling_average == "mean":
                    score[i] = torch.mean(
                        torch.tensor(self.rolling_average_buffers[i])
                    ).item()
                elif self.rolling_average == "median":
                    score[i] = torch.median(
                        torch.tensor(self.rolling_average_buffers[i])
                    ).item()
                else:
                    raise NotImplementedError(
                        f"Unrecognized rolling average: {self.rolling_average}"
                    )

        # Store the scores for potential retrieval (used by evaluator for histograms)
        self.last_scores_original = score_original
        self.last_scores_rolling_avg = (
            score if self.rolling_average is not None else None
        )

        return score

    def reset_rolling_average_buffer(self, index: int) -> None:
        """Reset the rolling average buffer for a given index. The index corresponds
        to the environment index.
        """
        if self.rolling_average is not None:
            self.rolling_average_buffers[index] = collections.deque(
                self.rolling_average_size * [float("-inf")], self.rolling_average_size
            )

    def update_params(self, params):
        self.params = dc(params)

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save(self.params, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        self.params = torch.load(load_path)

    def train_percentile_step(self, percentile: float) -> float:
        """Return threshold for a target step_afhp percentile.

        Uses per-step scores: the p-th percentile of all per-step scores.
        """
        if self._train_scores is None:
            raise ValueError(
                "Training scores not available. Call generate_scores() first."
            )
        return np.percentile(self._train_scores, percentile)

    def train_percentile_level(self, percentile: float) -> float:
        """Return threshold for a target level_afhp percentile.

        Uses per-episode max scores: the p-th percentile of episode-max-scores
        is the threshold where (100-p)% of episodes have at least one step
        exceeding it, i.e., level_afhp = (100-p)%.
        """
        if self._train_episode_max_scores is None:
            raise ValueError(
                "Episode-level training scores not available. "
                "Call generate_scores() first."
            )
        return np.percentile(self._train_episode_max_scores, percentile)

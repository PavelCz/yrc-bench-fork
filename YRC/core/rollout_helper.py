from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from YRC.core.configs.global_configs import get_global_variable
from torch.distributions.categorical import Categorical
import numpy as np
from YRC.core.utils import to_tensor


class RolloutHelper:
    def __init__(self, config, env):
        self.args = config.coord_policy
        if config.coord_policy.collect_data_agent == "weak":
            self.agent = env.weak_agent
        elif config.coord_policy.collect_data_agent == "strong":
            self.agent = env.strong_agent
        self.feature_type = config.coord_policy.feature_type
        self.explore_temp = 1.0

    def gather_rollouts(
        self,
        env,
        num_rollouts: int,
        gather_all=False,
        return_list=False,
        return_metadata=False,
    ) -> Union[
        torch.Tensor,
        List[torch.Tensor],
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]],
    ]:
        """
        Gathers rollouts from the environment.

        Args:
            env: Environment
            num_rollouts: Number of rollouts to gather
            gather_all: Whether to gather all rollouts. If set to false, only a random
                subset of 0.5% of the rollouts are gathered.
            return_list: If set to False, the list of observations is concatenated into
                a single contiguous tensor. If set to True, instead, the observations
                are returned as a list of tensors.
        """
        assert num_rollouts % env.num_envs == 0
        observations = []
        completed_level_seeds = []
        for i in range(num_rollouts // env.num_envs):
            rollout_result = self._rollout_once(
                env, gather_all=gather_all, return_metadata=return_metadata
            )
            if return_metadata:
                rollout_observations, rollout_metadata = rollout_result
                observations.extend(rollout_observations)
                completed_level_seeds.extend(rollout_metadata["completed_level_seeds"])
            else:
                observations.extend(rollout_result)
        if self.feature_type in ["hidden_obs", "hidden_dist", "obs_dist"]:
            feature_tensors = [[], []]
            for i, tensor in enumerate(observations):
                if isinstance(tensor, dict):
                    tensor = tensor["image"]
                feature_tensors[i % 2].append(tensor)
            observations = [torch.cat(tensors, dim=0) for tensors in feature_tensors]
        elif self.feature_type in ["obs_hidden_dist"]:
            feature_tensors = [[], [], []]
            for i, tensor in enumerate(observations):
                feature_tensors[i % 3].append(tensor)
            if get_global_variable("benchmark") == "procgen":
                observations = [
                    torch.cat(tensors, dim=0) for tensors in feature_tensors
                ]
            elif get_global_variable("benchmark") == "minigrid":
                observations = []
                for tensors in feature_tensors:
                    if isinstance(tensors[0], dict):
                        obs_dict = {}
                        for tensor_dict in tensors:
                            for k, v in tensor_dict.items():
                                obs_dict.setdefault(k, []).extend(v)
                        for v in obs_dict.values():
                            observations.append(
                                v if isinstance(v[0], str) else torch.stack(v, dim=0)
                            )
                    else:
                        observations.append(torch.cat(tensors, dim=0))
        else:
            if get_global_variable(
                "benchmark"
            ) == "minigrid" and self.feature_type not in [
                "hidden",
                "dist",
                "hidden_dist",
            ]:
                observations = observations[1::3]
                if not return_list:
                    # TODO: I (pavel) just changed this to observations from
                    # observations[1::3] I assume this was a copy paste bug previously.
                    # This is not a big deal, since we have no plans to use mingrid or
                    # other feature types.
                    observations = torch.cat(observations, dim=0)
            else:
                if not return_list:
                    observations = torch.stack(observations)
        if return_metadata:
            return observations, {"completed_level_seeds": completed_level_seeds}
        return observations

    def _rollout_once(self, env, gather_all=False, return_metadata=False):
        def sample_action(logit):
            """Samples an action using a categorical distribution with exploration temperature."""
            dist = Categorical(logits=logit / self.explore_temp)
            return dist.sample().cpu().numpy()

        def get_features(obs, feature_type):
            """Retrieves features based on the specified feature type."""
            feature_map = {
                "obs": lambda obs: (
                    obs["env_obs"]["image"]
                    if get_global_variable("benchmark") == "cliport"
                    else obs["env_obs"]
                ),
                "hidden": lambda obs: obs["weak_features"],
                "hidden_obs": lambda obs: (
                    [
                        obs["env_obs"]["image"],
                        obs["weak_features"],
                    ]
                    if get_global_variable("benchmark") == "cliport"
                    else [obs["env_obs"], obs["weak_features"]]
                ),
                "dist": lambda obs: obs["weak_logit"],
                "hidden_dist": lambda obs: [obs["weak_features"], obs["weak_logit"]],
                "obs_dist": lambda obs: (
                    [obs["env_obs"]["image"], obs["weak_logit"]]
                    if get_global_variable("benchmark") == "cliport"
                    else [obs["env_obs"], obs["weak_logit"]]
                ),
                "obs_hidden_dist": lambda obs: (
                    [
                        obs["env_obs"]["image"],
                        obs["weak_features"],
                        obs["weak_logit"],
                    ]
                    if get_global_variable("benchmark") == "cliport"
                    else [obs["env_obs"], obs["weak_features"], obs["weak_logit"]]
                ),
            }
            return feature_map[feature_type](obs)

        agent = self.agent
        agent.eval()
        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        observations = []
        completed_level_seeds = []

        while not has_done.all():
            logit = agent.forward(obs["env_obs"])

            if get_global_variable("benchmark") == "cliport":
                obs_features = get_features(obs, self.feature_type)
                obs_features = self.maybe_convert_to_tensor(obs_features)
                observations.extend(obs_features)
            else:
                for i in range(env.num_envs):
                    if not has_done[i]:
                        obs_features = get_features(obs, self.feature_type)
                        if (
                            gather_all or np.random.rand() < 0.005
                        ):  # Randomly sample for memory efficiency
                            obs_features = self.maybe_convert_to_tensor(obs_features)
                            if isinstance(obs_features, dict):
                                observations.extend(v for k, v in obs_features.items())
                            else:
                                observations.extend(obs_features)

            action = sample_action(logit)
            obs, reward, done, info = env.step(action)
            newly_done = np.asarray(done, dtype=bool) & ~has_done
            if return_metadata:
                completed_level_seeds.extend(
                    self._extract_completed_level_seeds(info, newly_done)
                )
            has_done |= done

        if return_metadata:
            return observations, {"completed_level_seeds": completed_level_seeds}
        return observations

    def _extract_completed_level_seeds(
        self, info: Any, newly_done: np.ndarray
    ) -> List[int]:
        completed_level_seeds = []
        for i, done in enumerate(newly_done):
            if not done:
                continue
            level_seed = self._get_level_seed(info, i)
            if level_seed is not None:
                completed_level_seeds.append(level_seed)
        return completed_level_seeds

    def _get_level_seed(self, info: Any, index: int) -> Optional[int]:
        if isinstance(info, list):
            info_i = info[index] if index < len(info) else None
            if not isinstance(info_i, dict):
                return None
            level_seed = info_i.get("prev_level_seed", info_i.get("level_seed"))
        elif isinstance(info, dict):
            level_seed = None
            for key in ("prev_level_seed", "level_seed"):
                if key not in info:
                    continue
                value = info[key]
                if hasattr(value, "__getitem__") and not isinstance(
                    value, (str, bytes)
                ):
                    try:
                        level_seed = value[index]
                    except (IndexError, KeyError, TypeError):
                        level_seed = value
                else:
                    level_seed = value
                break
        else:
            return None

        if level_seed is None:
            return None
        try:
            level_seed = int(level_seed)
        except (TypeError, ValueError):
            return None
        return None if level_seed < 0 else level_seed

    def maybe_convert_to_tensor(self, features):
        """Converts features to tensors if they are not already tensors."""
        if isinstance(
            features, list
        ):  # Handle lists of features (e.g., for concatenation)
            return [to_tensor(f) if not torch.is_tensor(f) else f for f in features]
        return to_tensor(features) if not torch.is_tensor(features) else features

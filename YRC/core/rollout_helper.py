from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from YRC.core.configs.global_configs import get_global_variable
from torch.distributions.categorical import Categorical
import numpy as np
from YRC.core.utils import to_tensor


class RolloutHelper:
    def __init__(self, config, env=None, agent=None):
        self.args = config.coord_policy
        if agent is not None:
            self.agent = agent
        elif config.coord_policy.collect_data_agent == "weak":
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
        chunk_size: Optional[int] = None,
        chunk_callback: Optional[Callable[[List[torch.Tensor]], None]] = None,
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
            chunk_size: If set with chunk_callback, flush observations when this many
                observations have accumulated.
            chunk_callback: Callback that receives flushed observation lists. This is
                intended for streaming Procgen observation rollouts to disk.
        """
        if num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
        if chunk_callback is not None:
            if not return_list:
                raise ValueError("chunk_callback requires return_list=True.")
            if self.feature_type != "obs":
                raise NotImplementedError(
                    "Chunked rollout flushing currently supports feature_type='obs' "
                    f"only, got {self.feature_type!r}."
                )
            if chunk_size is None or chunk_size <= 0:
                raise ValueError(
                    f"chunk_size must be positive when chunk_callback is set, got "
                    f"{chunk_size}"
                )

        observations = []
        completed_level_seeds = []
        num_completed = 0
        num_started = min(num_rollouts, env.num_envs)
        active_rollouts = np.zeros(env.num_envs, dtype=bool)
        active_rollouts[:num_started] = True

        agent = self.agent
        agent.eval()
        with torch.no_grad():
            obs = env.reset()

            while num_completed < num_rollouts:
                logit = agent.forward(obs["env_obs"])

                for i in range(env.num_envs):
                    if not active_rollouts[i]:
                        continue
                    obs_features = self._get_features_for_env(obs, self.feature_type, i)
                    if gather_all or np.random.rand() < 0.005:
                        obs_features = self.maybe_convert_to_tensor(obs_features)
                        if isinstance(obs_features, dict):
                            observations.extend(v for v in obs_features.values())
                        elif isinstance(obs_features, list):
                            observations.extend(obs_features)
                        else:
                            observations.append(obs_features)
                        if (
                            chunk_callback is not None
                            and chunk_size is not None
                            and len(observations) >= chunk_size
                        ):
                            chunk_callback(observations)
                            observations = []

                action = self._sample_action(logit)
                obs, reward, done, info = env.step(action)
                completed_this_step = np.asarray(done, dtype=bool) & active_rollouts

                if return_metadata:
                    completed_level_seeds.extend(
                        self._extract_completed_level_seeds(info, completed_this_step)
                    )

                completed_indices = np.flatnonzero(completed_this_step)
                num_completed += len(completed_indices)

                for i in completed_indices:
                    if num_started < num_rollouts:
                        num_started += 1
                    else:
                        active_rollouts[i] = False

        if chunk_callback is not None and observations:
            chunk_callback(observations)
            observations = []

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

    def gather_acting_policy_rollouts(
        self,
        env,
        num_rollouts: int,
        gather_all=False,
        return_metadata=False,
        chunk_size: Optional[int] = None,
        chunk_callback: Optional[Callable[[List[torch.Tensor]], None]] = None,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, Any]]]:
        """Gather observations by stepping a raw environment with an acting policy.

        This path bypasses CoordEnv, so it does not compute coordination features or
        reinterpret acting-policy actions as weak/strong switch actions.
        """
        if self.feature_type != "obs":
            raise NotImplementedError(
                "Direct acting-policy rollout collection currently supports "
                f"feature_type='obs' only, got {self.feature_type!r}."
            )
        if num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
        if chunk_callback is not None and (chunk_size is None or chunk_size <= 0):
            raise ValueError(
                f"chunk_size must be positive when chunk_callback is set, got "
                f"{chunk_size}"
            )

        observations = []
        completed_level_seeds = []
        num_completed = 0
        num_started = min(num_rollouts, env.num_envs)
        active_rollouts = np.zeros(env.num_envs, dtype=bool)
        active_rollouts[:num_started] = True

        agent = self.agent
        agent.eval()
        with torch.no_grad():
            obs = env.reset()
            if hasattr(agent, "reset"):
                agent.reset(np.array([True] * env.num_envs))

            while num_completed < num_rollouts:
                for i in range(env.num_envs):
                    if not active_rollouts[i]:
                        continue
                    if gather_all or np.random.rand() < 0.005:
                        observations.append(self.maybe_convert_to_tensor(obs[i]))
                        if (
                            chunk_callback is not None
                            and chunk_size is not None
                            and len(observations) >= chunk_size
                        ):
                            chunk_callback(observations)
                            observations = []

                action = agent.act(obs, greedy=False)
                obs, reward, done, info = env.step(action)
                completed_this_step = np.asarray(done, dtype=bool) & active_rollouts

                if hasattr(agent, "reset"):
                    agent.reset(done)

                if return_metadata:
                    completed_level_seeds.extend(
                        self._extract_completed_level_seeds(info, completed_this_step)
                    )

                completed_indices = np.flatnonzero(completed_this_step)
                num_completed += len(completed_indices)

                for i in completed_indices:
                    if num_started < num_rollouts:
                        num_started += 1
                    else:
                        active_rollouts[i] = False

        if chunk_callback is not None and observations:
            chunk_callback(observations)
            observations = []

        if return_metadata:
            return observations, {"completed_level_seeds": completed_level_seeds}
        return observations

    def _sample_action(self, logit):
        """Samples an action using a categorical distribution with exploration temperature."""
        dist = Categorical(logits=logit / self.explore_temp)
        return dist.sample().cpu().numpy()

    def _get_features(self, obs, feature_type):
        """Retrieves batched features based on the specified feature type."""
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

    def _get_features_for_env(self, obs, feature_type, env_idx):
        return self._slice_batch_item(self._get_features(obs, feature_type), env_idx)

    def _slice_batch_item(self, value, index):
        if isinstance(value, dict):
            return {k: self._slice_batch_item(v, index) for k, v in value.items()}
        if isinstance(value, list):
            return [self._slice_batch_item(v, index) for v in value]
        if isinstance(value, tuple):
            return tuple(self._slice_batch_item(v, index) for v in value)
        return value[index]

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

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from YRC.core.rollout_helper import RolloutHelper


class DummyAgent:
    def eval(self):
        return self

    def forward(self, env_obs):
        return torch.zeros((env_obs.shape[0], 2), dtype=torch.float32)


class DummyActingAgent:
    def __init__(self):
        self.observations = []
        self.reset_masks = []

    def eval(self):
        return self

    def reset(self, should_reset):
        self.reset_masks.append(np.asarray(should_reset).tolist())

    def act(self, obs, greedy=False):
        self.observations.append(obs.copy())
        return np.array([7, 8], dtype=np.int32)


class DummyEnv:
    def __init__(self):
        self.num_envs = 2
        self.weak_agent = DummyAgent()
        self._step_idx = 0

    def reset(self):
        self._step_idx = 0
        return {"env_obs": np.array([[1.0], [2.0]], dtype=np.float32)}

    def step(self, action):
        self._step_idx += 1
        obs = {"env_obs": np.array([[3.0], [4.0]], dtype=np.float32)}
        reward = np.zeros(self.num_envs, dtype=np.float32)
        if self._step_idx == 1:
            done = np.array([True, False], dtype=bool)
            info = [{"prev_level_seed": 101}, {}]
        else:
            done = np.array([True, True], dtype=bool)
            info = [{"prev_level_seed": 999}, {"prev_level_seed": 202}]
        return obs, reward, done, info


class DummySequentialEnv:
    def __init__(self):
        self.num_envs = 2
        self.weak_agent = DummyAgent()
        self._step_idx = 0

    def reset(self):
        self._step_idx = 0
        return {"env_obs": np.array([[10.0], [20.0]], dtype=np.float32)}

    def step(self, action):
        self._step_idx += 1
        reward = np.zeros(self.num_envs, dtype=np.float32)
        if self._step_idx == 1:
            obs = {"env_obs": np.array([[11.0], [21.0]], dtype=np.float32)}
            done = np.array([True, False], dtype=bool)
            info = [{"prev_level_seed": 101}, {}]
        else:
            obs = {"env_obs": np.array([[12.0], [22.0]], dtype=np.float32)}
            done = np.array([True, True], dtype=bool)
            info = [{"prev_level_seed": 303}, {"prev_level_seed": 202}]
        return obs, reward, done, info


class DummyRawSequentialEnv:
    def __init__(self):
        self.num_envs = 2
        self._step_idx = 0
        self.actions = []

    def reset(self):
        self._step_idx = 0
        return np.array([[10.0], [20.0]], dtype=np.float32)

    def step(self, action):
        self.actions.append(np.asarray(action).tolist())
        self._step_idx += 1
        reward = np.zeros(self.num_envs, dtype=np.float32)
        if self._step_idx == 1:
            obs = np.array([[11.0], [21.0]], dtype=np.float32)
            done = np.array([True, False], dtype=bool)
            info = [{"prev_level_seed": 101}, {}]
        else:
            obs = np.array([[12.0], [22.0]], dtype=np.float32)
            done = np.array([True, True], dtype=bool)
            info = [{"prev_level_seed": 303}, {"prev_level_seed": 202}]
        return obs, reward, done, info


def make_config():
    return SimpleNamespace(
        coord_policy=SimpleNamespace(
            collect_data_agent="weak",
            feature_type="obs",
        )
    )


def test_rollout_helper_records_only_first_completed_seed_per_env():
    env = DummyEnv()
    helper = RolloutHelper(make_config(), env)

    with patch("YRC.core.rollout_helper.get_global_variable", return_value="procgen"):
        _, metadata = helper.gather_rollouts(
            env,
            num_rollouts=2,
            gather_all=True,
            return_list=True,
            return_metadata=True,
        )

    assert metadata["completed_level_seeds"] == [101, 202]
    assert metadata["completed_rollout_observation_counts"] == [1, 2]


def test_rollout_helper_supports_arbitrary_rollout_counts():
    env = DummySequentialEnv()
    helper = RolloutHelper(make_config(), env)

    with patch("YRC.core.rollout_helper.get_global_variable", return_value="procgen"):
        observations, metadata = helper.gather_rollouts(
            env,
            num_rollouts=3,
            gather_all=True,
            return_list=True,
            return_metadata=True,
        )

    assert len(observations) == 4
    assert metadata["completed_level_seeds"] == [101, 303, 202]
    assert metadata["completed_rollout_observation_counts"] == [1, 1, 2]


def test_rollout_helper_flushes_observation_chunks():
    env = DummySequentialEnv()
    helper = RolloutHelper(make_config(), env)
    chunks = []

    def record_chunk(observations):
        chunks.append([obs.clone() for obs in observations])

    with patch("YRC.core.rollout_helper.get_global_variable", return_value="procgen"):
        observations, metadata = helper.gather_rollouts(
            env,
            num_rollouts=3,
            gather_all=True,
            return_list=True,
            return_metadata=True,
            chunk_size=2,
            chunk_callback=record_chunk,
        )

    assert observations == []
    assert [[obs.item() for obs in chunk] for chunk in chunks] == [
        [10.0, 20.0],
        [11.0, 21.0],
    ]
    assert metadata["completed_level_seeds"] == [101, 303, 202]
    assert metadata["completed_rollout_observation_counts"] == [1, 1, 2]


def test_rollout_helper_steps_raw_env_with_acting_policy_actions():
    env = DummyRawSequentialEnv()
    agent = DummyActingAgent()
    helper = RolloutHelper(make_config(), env, agent=agent)

    observations, metadata = helper.gather_acting_policy_rollouts(
        env,
        num_rollouts=3,
        gather_all=True,
        return_metadata=True,
    )

    assert [obs.item() for obs in observations] == [10.0, 20.0, 11.0, 21.0]
    assert metadata["completed_level_seeds"] == [101, 303, 202]
    assert metadata["completed_rollout_observation_counts"] == [1, 1, 2]
    assert env.actions == [[7, 8], [7, 8]]
    assert len(agent.observations) == 2
    assert agent.reset_masks == [[True, True], [True, False], [True, True]]

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

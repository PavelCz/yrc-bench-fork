from types import SimpleNamespace

import numpy as np
import torch

from YRC.core.configs.global_configs import set_global_variable
from YRC.policies.threshold import ThresholdPolicy


class FakeAgent:
    def eval(self):
        return self

    def forward(self, obs):
        batch_size = obs.shape[0]
        return torch.tensor([[2.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)


class FakeCalibrationEnv:
    num_envs = 2
    env_obs = None

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        return self._obs()

    def step(self, actions):
        self.step_count += 1
        done_schedule = {
            1: [False, True],
            2: [True, False],
            3: [False, True],
        }
        done = np.array(done_schedule[self.step_count], dtype=bool)
        info = [{}, {}]
        return self._obs(), np.zeros(self.num_envs), done, info

    def _obs(self):
        return {"env_obs": torch.zeros((self.num_envs, 3, 64, 64))}


class FakePolicyEnv:
    num_envs = 2
    weak_agent = FakeAgent()


def make_policy():
    set_global_variable("benchmark", "procgen")
    set_global_variable("device", torch.device("cpu"))
    config = SimpleNamespace(
        coord_policy=SimpleNamespace(
            metric="max_prob",
            rolling_average="none",
            rolling_average_size=10,
        )
    )
    return ThresholdPolicy(config, FakePolicyEnv())


def test_threshold_generate_scores_uses_single_sequential_rollout():
    policy = make_policy()
    env = FakeCalibrationEnv()

    scores = policy.generate_scores(env, num_rollouts=3)

    assert env.reset_count == 1
    assert env.step_count == 3
    assert len(scores) == 6
    assert policy._train_episode_max_scores.shape == (3,)

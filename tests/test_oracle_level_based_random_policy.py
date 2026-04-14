import unittest
from unittest.mock import patch

import torch

from YRC.policies.base import OracleLevelBasedRandomPolicy


class DummyEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


class OracleLevelBasedRandomPolicyTests(unittest.TestCase):
    def test_delays_decision_and_latches_within_episode(self):
        policy = OracleLevelBasedRandomPolicy(config=None, env=DummyEnv(num_envs=2))
        policy.update_params(0.5)

        with patch.object(torch, "rand", lambda *args, **kwargs: torch.tensor([0.25])):
            first_action = policy.act(
                {"episode_timestep": [0, 0], "level_ood_gt": [True, False]}
            )
            self.assertEqual(first_action.tolist(), [0, 0])

            second_action = policy.act(
                {"episode_timestep": [1, 1], "level_ood_gt": [True, False]}
            )
            self.assertEqual(second_action.tolist(), [1, 0])

            latched_action = policy.act(
                {"episode_timestep": [2, 2], "level_ood_gt": [False, True]}
            )
            self.assertEqual(latched_action.tolist(), [1, 0])

    def test_control_above_one_always_asks_on_ood_and_can_ask_on_id(self):
        policy = OracleLevelBasedRandomPolicy(config=None, env=DummyEnv(num_envs=2))
        policy.update_params(1.25)

        with patch.object(torch, "rand", lambda *args, **kwargs: torch.tensor([0.20])):
            action = policy.act(
                {"episode_timestep": [1, 1], "level_ood_gt": [True, False]}
            )
            self.assertEqual(action.tolist(), [1, 1])

    def test_level_percentile_mapping_covers_full_control_range(self):
        policy = OracleLevelBasedRandomPolicy(config=None, env=DummyEnv(num_envs=1))

        self.assertAlmostEqual(policy.train_percentile_level(100.0), 0.0)
        self.assertAlmostEqual(policy.train_percentile_level(50.0), 1.0)
        self.assertAlmostEqual(policy.train_percentile_level(0.0), 2.0)

        with self.assertRaises(NotImplementedError):
            policy.train_percentile_step(50.0)


if __name__ == "__main__":
    unittest.main()

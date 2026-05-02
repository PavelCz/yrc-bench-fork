import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "lib" / "train-procgen-pytorch")
)

from lib.procgen.procgen import env as procgen_env  # noqa: E402

from common.logger import Logger  # noqa: E402
from agents.ppo import PPO  # noqa: E402


def test_procgen_gym3_env_passes_randomize_agent_start_option(monkeypatch):
    captured = {}

    def fake_init(self, num, env_name, options, **kwargs):
        captured["num"] = num
        captured["env_name"] = env_name
        captured["options"] = options
        captured["kwargs"] = kwargs

    monkeypatch.setattr(procgen_env.BaseProcgenEnv, "__init__", fake_init)

    procgen_env.ProcgenGym3Env(
        num=2,
        env_name="maze_afh",
        randomize_agent_start=True,
        num_levels=1,
        start_level=123,
    )

    assert captured["num"] == 2
    assert captured["env_name"] == "maze_afh"
    assert captured["options"]["randomize_agent_start"] is True
    assert captured["kwargs"]["num_levels"] == 1
    assert captured["kwargs"]["start_level"] == 123


def test_procgen_gym3_env_defaults_randomize_agent_start_to_false(monkeypatch):
    captured = {}

    def fake_init(self, num, env_name, options, **kwargs):
        del num, env_name, kwargs
        captured["options"] = options

    monkeypatch.setattr(procgen_env.BaseProcgenEnv, "__init__", fake_init)

    procgen_env.ProcgenGym3Env(num=1, env_name="maze_afh")

    assert captured["options"]["randomize_agent_start"] is False


def test_logger_records_random_start_validation_stream(tmp_path):
    logger = Logger(
        n_envs=1,
        logdir=str(tmp_path),
        use_random_start_validation=True,
    )

    rew_batch = np.array([[1.0], [2.0]])
    done_batch = np.array([[False], [True]])
    rew_batch_v = np.array([[3.0], [4.0]])
    done_batch_v = np.array([[False], [True]])
    rew_batch_v_random_start = np.array([[5.0], [6.0]])
    done_batch_v_random_start = np.array([[False], [True]])

    logger.feed(
        rew_batch,
        done_batch,
        rew_batch_v,
        done_batch_v,
        rew_batch_v_random_start,
        done_batch_v_random_start,
    )

    stats = logger._get_episode_statistics()

    assert stats["Rewards/mean_episodes"] == 3.0
    assert stats["[Valid] Rewards/mean_episodes"] == 7.0
    assert stats["[Valid Random Start] Rewards/mean_episodes"] == 11.0


def test_logger_records_direct_validation_episode_stats(tmp_path):
    logger = Logger(
        n_envs=1,
        logdir=str(tmp_path),
        use_random_start_validation=True,
    )

    logger.feed_validation(
        episode_returns=[10.0, 0.0, 10.0],
        episode_lengths=[12, 500, 30],
        episode_timeouts=[0, 1, 0],
        random_start=False,
    )
    logger.feed_validation(
        episode_returns=[10.0, 10.0],
        episode_lengths=[15, 20],
        random_start=True,
    )

    stats = logger._get_episode_statistics()

    assert stats["[Valid] Rewards/mean_episodes"] == np.mean([10.0, 0.0, 10.0])
    assert stats["[Valid] Len/mean_episodes"] == np.mean([12, 500, 30])
    assert stats["[Valid] Len/mean_timeout"] == np.mean([0, 1, 0])
    assert stats["[Valid Random Start] Rewards/mean_episodes"] == 10.0
    assert stats["[Valid Random Start] Len/mean_episodes"] == np.mean([15, 20])


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


class DeterministicPPO(PPO):
    def predict(self, obs, hidden_state, done):
        del done
        n_envs = len(obs)
        return (
            np.zeros(n_envs, dtype=np.int64),
            np.zeros(n_envs),
            np.zeros(n_envs),
            hidden_state,
        )


class FakeValidationEnv:
    def __init__(self, steps):
        self.steps = list(steps)
        self.step_idx = 0

    def step(self, act):
        del act
        rewards, dones, raw_rewards = self.steps[self.step_idx]
        self.step_idx += 1
        obs = np.zeros((len(rewards), 1), dtype=np.float32)
        infos = [{"env_reward": raw_reward} for raw_reward in raw_rewards]
        return (
            obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )


def test_ppo_validation_collects_complete_raw_episode_returns():
    agent = DeterministicPPO(
        env=None,
        policy=DummyPolicy(),
        logger=None,
        storage=None,
        device=torch.device("cpu"),
        n_checkpoints=1,
        n_envs=2,
        num_validation_episodes=3,
    )
    env = FakeValidationEnv(
        steps=[
            ([0.1, 0.2], [False, False], [0.0, 0.0]),
            ([0.1, 0.2], [True, False], [10.0, 0.0]),
            ([0.1, 0.2], [False, True], [0.0, 10.0]),
            ([0.1, 0.2], [True, False], [10.0, 0.0]),
        ]
    )
    obs = np.zeros((2, 1), dtype=np.float32)
    hidden_state = np.zeros((2, 1), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    _, _, _, stats = agent.run_validation(env, obs, hidden_state, done)

    assert stats["episode_returns"] == [10.0, 10.0, 10.0]
    assert stats["episode_lengths"] == [2, 3, 2]
    assert env.step_idx == 4

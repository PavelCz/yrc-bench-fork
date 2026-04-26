import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "lib" / "train-procgen-pytorch")
)

from lib.procgen.procgen import env as procgen_env  # noqa: E402

from common.logger import Logger  # noqa: E402


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

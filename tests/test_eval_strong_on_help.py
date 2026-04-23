import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

if importlib.util.find_spec("gymnasium") is None:
    import gym
else:
    import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_strong_on_help as reval
from eval_strong_on_help import (
    PointRecord,
    build_full_budget_point_result,
    build_strong_reval_point,
    default_procgen_timeout,
    extract_afhp_from_summary,
    extract_point_record,
    full_budget_output_path,
    load_eval_seeds,
    load_reval_config,
    make_env_config,
    resolve_afhp_metric,
    strong_reval_output_path,
)
from lib.procgen.procgen.env import BaseProcgenEnv
from YRC.core.configs.config import ConfigDict
from YRC.core.environment import CoordEnv


def test_extract_point_record_normalizes_help_fields():
    point = {
        "threshold": 0.25,
        "summary": {
            "test": {
                "level_seeds": np.array([101, 202]),
                "level_ood_pred": np.array([False, True]),
                "first_ood_timestep": np.array([None, 7], dtype=object),
                "raw_returns": np.array([1.0, 2.5]),
            }
        },
    }

    record = extract_point_record(point, index=3)

    assert record.index == 3
    assert record.threshold == 0.25
    assert record.level_seeds == [101, 202]
    assert record.level_ood_pred == [False, True]
    assert record.first_help_timesteps == [None, 7]
    assert record.raw_returns == [1.0, 2.5]


def test_load_reval_config_prefers_sibling_config_without_cli_overrides(
    tmp_path,
    monkeypatch,
):
    npz_path = tmp_path / "eval_results.npz"
    sibling_config = tmp_path / "config.json"
    sibling_config.write_text(
        json.dumps(
            {
                "general": {"device": "cuda:0"},
                "evaluation": {"defer_to_oracle": True},
                "eval_mode": True,
                "overwrite": False,
                "use_wandb": True,
            }
        )
    )
    calls = []

    def fake_load(payload, flags=None):
        calls.append((payload, flags))
        return payload

    monkeypatch.setattr(reval.config_utils, "load", fake_load)
    args = SimpleNamespace(
        config="fallback.yaml",
        eval_mode=True,
        overwrite=False,
        use_wandb=False,
    )

    load_reval_config(args, npz_path)

    loaded_config = json.loads(calls[0][0])
    assert calls[0][1] is None
    assert loaded_config["general"]["device"] == 0
    assert loaded_config["evaluation"]["defer_to_oracle"] is True
    assert loaded_config["eval_mode"] is False
    assert loaded_config["overwrite"] is True
    assert loaded_config["use_wandb"] is False


def test_make_env_config_clones_configdict_without_deepcopy_protocol():
    original_env_config = ConfigDict(
        common={
            "num_envs": 4,
            "num_threads": 4,
            "max_steps": None,
            "env_name": "coinrun",
        },
        train={"num_levels": 1},
    )
    config = SimpleNamespace(environment=original_env_config)

    cloned_env_config = make_env_config(config, num_envs=1, max_steps=1004)

    assert cloned_env_config.common.num_envs == 1
    assert cloned_env_config.common.num_threads == 1
    assert cloned_env_config.common.max_steps == 1004
    assert original_env_config.common.num_envs == 4
    assert original_env_config.common.num_threads == 4
    assert original_env_config.common.max_steps is None


def test_build_strong_reval_point_no_help_returns_nan_values():
    record = PointRecord(
        index=0,
        threshold=1.0,
        split="test",
        level_seeds=[101, 202],
        level_ood_pred=[False, False],
        first_help_timesteps=[None, None],
        raw_returns=[1.0, 2.0],
    )

    comparison = build_strong_reval_point(record, {})

    assert comparison["help_seeds"] == []
    assert np.isnan(comparison["original_help_performance"])
    assert np.isnan(comparison["strong_performance"])


def test_build_strong_reval_point_aligns_help_seed_outputs():
    record = PointRecord(
        index=2,
        threshold=0.3,
        split="test",
        level_seeds=[101, 202, 303],
        level_ood_pred=[False, True, True],
        first_help_timesteps=[None, 5, 1],
        raw_returns=[1.0, 2.0, 4.0],
    )

    comparison = build_strong_reval_point(
        record,
        strong_seed_to_return={202: 8.0, 303: 10.0},
    )

    assert comparison["help_seeds"] == [202, 303]
    assert comparison["original_help_performance"] == 3.0
    assert comparison["strong_performance"] == 9.0


def test_build_full_budget_point_result_uses_env_return_mean_and_threshold():
    record = PointRecord(
        index=1,
        threshold=0.42,
        split="test",
        level_seeds=[1],
        level_ood_pred=[True],
        first_help_timesteps=[3],
        raw_returns=[2.0],
    )
    summary = {
        "action_1_frac": 0.25,
        "level_afhp": 0.5,
        "return_mean": 1.0,
        "env_return_mean": 8.0,
    }

    result = build_full_budget_point_result(record, summary, "step_afhp")

    assert result["afhp"] == 25.0
    assert result["performance"] == 8.0
    assert result["meta"]["threshold"] == 0.42
    assert result["meta"]["point_idx"] == 1


def test_extract_afhp_from_summary_matches_requested_metric():
    summary = {"action_1_frac": 0.125, "level_afhp": 0.75}

    assert extract_afhp_from_summary(summary, "step_afhp") == 12.5
    assert extract_afhp_from_summary(summary, "level_afhp") == 75.0


def test_resolve_afhp_metric_uses_threshold_sampler():
    config = SimpleNamespace(evaluation=SimpleNamespace(threshold_sampler="step_afhp"))
    assert resolve_afhp_metric(config) == "step_afhp"


def test_load_eval_seeds_uses_ood_eval_split(tmp_path):
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text(
        json.dumps(
            {
                "seeds": {
                    "validation": [1, 2],
                    "ood_eval": [100206, 101041, 100888],
                }
            }
        )
    )
    config = SimpleNamespace(
        environment=SimpleNamespace(level_seeds_file=str(seeds_file))
    )

    assert load_eval_seeds(config) == [100206, 101041, 100888]


def test_output_path_helpers_keep_split_artifacts_separate(tmp_path):
    npz_path = tmp_path / "results_test.npz"

    assert strong_reval_output_path(npz_path).name == "results_test_strong_reval.npz"
    assert full_budget_output_path(npz_path).name == "results_test_full_budget_eval.npz"


def test_default_procgen_timeout_matches_coinrun_and_maze_defaults():
    assert default_procgen_timeout("coinrun") == 1000
    assert default_procgen_timeout("maze_afh") == 500


def test_base_procgen_env_reset_remaining_timeout_calls_c_func():
    env = object.__new__(BaseProcgenEnv)
    env.num = 3
    calls = []

    def fake_call_c_func(name, env_idx, remaining_steps):
        calls.append((name, env_idx, remaining_steps))

    env.call_c_func = fake_call_c_func

    env.reset_remaining_timeout(2, 500)

    assert calls == [("reset_remaining_timeout", 2, 500)]


class FakeBaseEnv:
    def __init__(self):
        self.num_envs = 1
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.obs_shape = self.observation_space.shape
        self.reset_timeout_calls = []
        self.episode_step = 0

    def reset(self):
        self.episode_step = 0
        return np.zeros((1, 1), dtype=np.float32)

    def reset_remaining_timeout(self, env_idx, remaining_steps):
        self.reset_timeout_calls.append((env_idx, remaining_steps, self.episode_step))

    def step(self, env_action):
        self.episode_step += 1
        done = np.array([self.episode_step >= 2], dtype=bool)
        obs = np.full((1, 1), float(self.episode_step), dtype=np.float32)
        reward = np.array([1.0], dtype=np.float32)
        info = [{"env_reward": 1.0, "env_action": int(env_action[0])}]
        if done[0]:
            self.episode_step = 0
        return obs, reward, done, info

    def close(self):
        return None


class FakeAgent:
    hidden_dim = 2
    model = SimpleNamespace(logit_dim=3)

    def __init__(self, env_action):
        self.env_action = env_action

    def reset(self, done):
        del done

    def act(self, obs, greedy=False):
        del greedy
        batch = len(obs)
        return np.full(batch, self.env_action, dtype=np.int64)

    def get_hidden(self, obs):
        return torch.zeros((len(obs), self.hidden_dim), dtype=torch.float32)

    def forward(self, obs):
        return torch.zeros((len(obs), self.model.logit_dim), dtype=torch.float32)


def test_coord_env_timeout_reset_fires_once_per_episode():
    config = SimpleNamespace(
        act_greedy=False,
        strong_query_cost_ratio=0.0,
        switch_agent_cost_ratio=0.0,
    )
    base_env = FakeBaseEnv()
    env = CoordEnv(config, base_env, FakeAgent(1), FakeAgent(2))
    env.set_costs({"episode_length_mean": 10.0, "reward_mean": 10.0})
    env.enable_timeout_reset(500)

    env.reset()
    env.step(np.array([env.STRONG], dtype=np.int64))
    env.step(np.array([env.STRONG], dtype=np.int64))
    env.step(np.array([env.STRONG], dtype=np.int64))

    assert base_env.reset_timeout_calls == [(0, 500, 0), (0, 500, 0)]

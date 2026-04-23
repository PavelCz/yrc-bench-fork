import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_strong_on_help as reval
from eval_strong_on_help import (
    PointRecord,
    SanityRolloutResult,
    build_point_comparison,
    default_procgen_timeout,
    extract_point_record,
    dispatch_seeds_for_record,
    replay_actions_then_expert,
    reset_timeout_cap,
    load_reval_config,
    load_test_dispatch_seeds,
    make_env_config,
    rollout_coordination_sanity,
    validate_sanity_matches,
)
from YRC.core.configs.config import ConfigDict


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


def test_build_point_comparison_no_help_returns_nan_values():
    record = PointRecord(
        index=0,
        threshold=1.0,
        split="test",
        level_seeds=[101, 202],
        level_ood_pred=[False, False],
        first_help_timesteps=[None, None],
        raw_returns=[1.0, 2.0],
    )

    comparison = build_point_comparison(record, {}, {}, {})

    assert np.isnan(comparison["original_help_performance"])
    assert np.isnan(comparison["sanity_performance"])
    assert np.isnan(comparison["strong_performance"])
    assert np.isnan(comparison["reset_timeout_performance"])
    assert comparison["comparison_meta"]["help_requested"] is False
    assert comparison["comparison_meta"]["help_seeds"] == []


def test_build_point_comparison_aligns_help_seed_outputs():
    record = PointRecord(
        index=2,
        threshold=0.3,
        split="test",
        level_seeds=[101, 202, 303],
        level_ood_pred=[False, True, True],
        first_help_timesteps=[None, 5, 1],
        raw_returns=[1.0, 2.0, 4.0],
    )

    comparison = build_point_comparison(
        record,
        sanity_seed_to_return={202: 2.0, 303: 4.0},
        strong_seed_to_return={202: 8.0, 303: 10.0},
        reset_seed_results={
            202: {"return": 9.0, "timeout_cap": 504},
            303: {"return": 11.0, "timeout_cap": 500},
        },
    )

    assert comparison["original_help_performance"] == 3.0
    assert comparison["sanity_performance"] == 3.0
    assert comparison["strong_performance"] == 9.0
    assert comparison["reset_timeout_performance"] == 10.0
    assert comparison["comparison_meta"]["help_seeds"] == [202, 303]
    assert comparison["comparison_meta"]["reset_timeout_caps"] == [504, 500]


def test_load_test_dispatch_seeds_uses_ood_eval_split(tmp_path):
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

    assert load_test_dispatch_seeds(config) == [100206, 101041, 100888]


def test_dispatch_seeds_for_record_preserves_original_seed_file_order():
    record = PointRecord(
        index=0,
        threshold=0.5,
        split="test",
        level_seeds=[30, 10, 20],
        level_ood_pred=[False, True, False],
        first_help_timesteps=[None, 3, None],
        raw_returns=[3.0, 1.0, 2.0],
    )

    dispatch = dispatch_seeds_for_record(record, [10, 20, 30, 40])

    assert dispatch == [10, 20, 30]


def test_validate_sanity_matches_accepts_reordered_completion():
    record = PointRecord(
        index=0,
        threshold=0.5,
        split="test",
        level_seeds=[101, 202],
        level_ood_pred=[False, True],
        first_help_timesteps=[None, 3],
        raw_returns=[1.0, 2.0],
    )
    sanity = SanityRolloutResult(
        level_seeds=[202, 101],
        level_ood_pred=[True, False],
        first_help_timesteps=[3, None],
        raw_returns=[2.0, 1.0],
        pre_help_actions_by_seed={202: [0, 1], 101: [2]},
    )

    validate_sanity_matches(record, sanity)


def test_timeout_defaults_and_reset_cap_include_help_action():
    assert default_procgen_timeout("coinrun") == 1000
    assert default_procgen_timeout("maze_afh") == 500
    assert reset_timeout_cap(pre_help_steps=4, base_timeout=500) == 504


def test_validate_sanity_matches_rejects_changed_help_timestep():
    record = PointRecord(
        index=1,
        threshold=0.5,
        split="test",
        level_seeds=[101],
        level_ood_pred=[True],
        first_help_timesteps=[3],
        raw_returns=[1.0],
    )
    sanity = SanityRolloutResult(
        level_seeds=[101],
        level_ood_pred=[True],
        first_help_timesteps=[4],
        raw_returns=[1.0],
        pre_help_actions_by_seed={101: [0, 1, 2]},
    )

    with pytest.raises(AssertionError, match="first help timestep changed"):
        validate_sanity_matches(record, sanity)


class FakeReplayEnv:
    def __init__(self):
        self.num_envs = 1
        self.actions = []
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.array([[0.0]], dtype=np.float32)

    def step(self, action):
        action_value = int(np.asarray(action)[0])
        self.actions.append(action_value)
        self.step_count += 1
        obs = np.array([[float(self.step_count)]], dtype=np.float32)
        reward = np.array([float(action_value)], dtype=np.float32)
        done = np.array([self.step_count >= 3])
        info = [{"prev_level_seed": 123}] if done[0] else [{"level_seed": 123}]
        return obs, reward, done, info


class FakeStrongPolicy:
    def act(self, obs, greedy=False):
        return np.array([9], dtype=np.int64)


def test_replay_actions_then_expert_hands_off_at_help_point():
    env = FakeReplayEnv()

    result = replay_actions_then_expert(
        env,
        FakeStrongPolicy(),
        pre_help_actions=[1, 2],
        greedy=True,
    )

    assert env.actions == [1, 2, 9]
    assert result["return"] == 12.0
    assert result["level_seed"] == 123


class FakeCoordEnv:
    STRONG = 1

    def __init__(self):
        self.num_envs = 1
        self.actions = []
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {"env_obs": np.array([[0.0]], dtype=np.float32)}

    def step(self, action):
        action_value = int(np.asarray(action)[0])
        self.actions.append(action_value)
        self.step_count += 1
        obs = {"env_obs": np.array([[float(self.step_count)]], dtype=np.float32)}
        reward = np.array([float(action_value)], dtype=np.float32)
        done = np.array([self.step_count >= 2])
        info = [
            {
                "env_action": action_value + 10,
                "level_seed": 456,
                "prev_level_seed": 456 if done[0] else -1,
                "randomize_goal": False,
            }
        ]
        return obs, reward, done, info


class FakeCoordPolicy:
    def __init__(self):
        self.calls = 0

    def eval(self):
        pass

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        del obs, greedy
        action = np.array([1 if self.calls == 0 else 0], dtype=np.int64)
        self.calls += 1
        if return_scores_and_recons:
            return action, None, None
        return action


def test_sanity_rollout_defer_to_oracle_keeps_strong_after_first_help(
    monkeypatch,
):
    monkeypatch.setattr(reval, "update_policy_params", lambda policy, threshold: None)
    record = PointRecord(
        index=0,
        threshold=0.5,
        split="test",
        level_seeds=[456],
        level_ood_pred=[True],
        first_help_timesteps=[1],
        raw_returns=[2.0],
    )
    env = FakeCoordEnv()

    sanity = rollout_coordination_sanity(
        FakeCoordPolicy(),
        env,
        record,
        defer_to_oracle=True,
    )

    assert env.actions == [1, 1]
    assert sanity.level_seeds == [456]
    assert sanity.level_ood_pred == [True]
    assert sanity.first_help_timesteps == [1]
    assert sanity.raw_returns == [2.0]

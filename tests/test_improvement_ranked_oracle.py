import json
from types import SimpleNamespace

import numpy as np
import pytest

from YRC.core.evaluator import _info_level_seed
from YRC.policies.improvement_oracle import (
    ImprovementRankedOraclePolicy,
    improvement_by_seed,
    load_improvement_table,
    mean_return_by_seed,
    rank_seeds,
    save_improvement_table,
    should_ask_for_seed,
    table_from_policy_eval_results,
    unit_interval_from_seed,
)


class DummyEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


def test_mean_return_by_seed_averages_duplicates():
    averaged = mean_return_by_seed([1, 1, 2], [1.0, 3.0, 5.0])
    assert averaged == {1: 2.0, 2: 5.0}


def test_improvement_by_seed_uses_shared_seeds_only():
    improvements = improvement_by_seed({1: 1.0, 2: 4.0}, {2: 6.0, 3: 9.0})
    assert improvements == {2: 2.0}


def test_rank_seeds_orders_by_improvement_then_seed():
    ranked, rank_index = rank_seeds({3: 1.0, 1: 2.0, 2: 2.0})
    assert ranked == [1, 2, 3]
    assert rank_index == {1: 0, 2: 1, 3: 2}


def test_should_ask_uses_top_fraction_and_missing_seed_rule():
    _, rank_index = rank_seeds({10: 5.0, 20: 1.0, 30: 0.0})
    assert should_ask_for_seed(10, rank_index, 3, 0.0) is False
    assert should_ask_for_seed(10, rank_index, 3, 0.4) is True
    assert should_ask_for_seed(30, rank_index, 3, 0.4) is False
    assert should_ask_for_seed(99, rank_index, 3, 0.9) is False
    assert should_ask_for_seed(99, rank_index, 3, 1.0) is True


def test_unit_interval_from_seed_is_stable_and_in_unit_interval():
    first = unit_interval_from_seed(7)
    second = unit_interval_from_seed(7)
    assert first == second
    assert 0.0 <= first < 1.0


def test_table_roundtrip_and_bare_map(tmp_path):
    path = tmp_path / "table.json"
    save_improvement_table(
        path, {4: 1.5}, weak_returns={4: 2.0}, strong_returns={4: 3.5}
    )
    loaded = load_improvement_table(path)
    assert loaded["improvements"] == {4: 1.5}
    assert loaded["weak_returns"] == {4: 2.0}
    assert loaded["strong_returns"] == {4: 3.5}

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"8": 0.25}))
    assert load_improvement_table(bare)["improvements"] == {8: 0.25}


def test_table_from_policy_eval_results_pairs_by_seed():
    table = table_from_policy_eval_results(
        {"level_seeds": [1, 2], "all_returns": [1.0, 2.0]},
        {"level_seeds": [2, 1], "all_returns": [5.0, 2.0]},
    )
    assert table["improvements"] == {1: 1.0, 2: 3.0}


def test_policy_latches_after_first_step():
    policy = ImprovementRankedOraclePolicy(config=None, env=DummyEnv(num_envs=2))
    policy.set_table({11: 4.0, 22: 0.0})
    policy.update_params(0.5)

    first = policy.act({"episode_timestep": [0, 0], "level_seed": [-1, -1]})
    assert first.tolist() == [0, 0]

    second = policy.act({"episode_timestep": [1, 1], "level_seed": [11, 22]})
    assert second.tolist() == [1, 0]

    latched = policy.act({"episode_timestep": [2, 2], "level_seed": [22, 11]})
    assert latched.tolist() == [1, 0]


def test_policy_loads_table_from_config(tmp_path):
    table_path = tmp_path / "improvements.json"
    save_improvement_table(table_path, {5: 2.0, 6: -1.0})
    config = SimpleNamespace(
        coord_policy=SimpleNamespace(improvement_table=str(table_path))
    )
    policy = ImprovementRankedOraclePolicy(config=config, env=DummyEnv(num_envs=1))
    policy.update_params(0.5)
    action = policy.act({"episode_timestep": [1], "level_seed": [5]})
    assert action.tolist() == [1]


def test_percentile_mapping_is_linear_help_fraction():
    policy = ImprovementRankedOraclePolicy(config=None, env=DummyEnv(num_envs=1))
    assert policy.train_percentile_level(100.0) == pytest.approx(0.0)
    assert policy.train_percentile_level(50.0) == pytest.approx(0.5)
    assert policy.train_percentile_level(0.0) == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        policy.train_percentile_step(50.0)


def test_info_level_seed_prefers_current_then_prev():
    assert _info_level_seed({"level_seed": 12}) == 12
    assert _info_level_seed({"prev_level_seed": 9}) == 9
    assert _info_level_seed({}) == -1


def test_expected_afhp_matches_help_fraction_over_many_seeds():
    improvements = {seed: float(1000 - seed) for seed in range(200)}
    _, rank_index = rank_seeds(improvements)
    asked = [should_ask_for_seed(seed, rank_index, 200, 0.25) for seed in improvements]
    assert np.mean(asked) == pytest.approx(0.25, abs=0.02)

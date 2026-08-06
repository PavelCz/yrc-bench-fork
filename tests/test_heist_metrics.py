import pytest
import numpy as np
from procgen import ProcgenEnv

from YRC.envs.procgen.heist_metrics import (
    append_heist_episode_data,
    build_heist_metric_summary,
    compute_oracle_regret,
    extract_heist_episode_data,
    new_heist_episode_data,
    summarize_heist_metric_split,
    timeout_fraction,
)


def terminal_info(**overrides):
    info = {
        "prev_level/keys_collected": 4,
        "prev_level/num_keys": 6,
        "prev_level/total_chests": 3,
        "prev_level/chests_opened": 3,
        "prev_level/total_steps": 27,
        "prev_level_complete": 1,
    }
    info.update(overrides)
    return info


def test_extract_and_append_heist_episode_data():
    extracted = extract_heist_episode_data(terminal_info())

    assert extracted == {
        "keys_collected": 4,
        "num_keys": 6,
        "total_chests": 3,
        "chests_opened": 3,
        "total_steps": 27,
        "level_complete": True,
    }

    episode_data = new_heist_episode_data()
    append_heist_episode_data(episode_data, terminal_info())
    assert episode_data == {key: [value] for key, value in extracted.items()}


def test_extract_heist_episode_data_reports_all_missing_fields():
    with pytest.raises(KeyError, match="prev_level/chests_opened") as exc_info:
        extract_heist_episode_data({"prev_level/keys_collected": 1})

    assert "prev_level_complete" in str(exc_info.value)


def test_oracle_regret_uses_achievable_chest_cap():
    assert compute_oracle_regret(2.0, num_keys=2, total_chests=4) == 0.0
    assert compute_oracle_regret(2.0, num_keys=4, total_chests=2) == 0.0
    assert compute_oracle_regret(1.0, num_keys=4, total_chests=2) == 0.5


@pytest.mark.parametrize(
    ("num_keys", "total_chests"),
    [(0, 3), (3, 0), (-1, 3)],
)
def test_oracle_regret_rejects_non_positive_cap(num_keys, total_chests):
    with pytest.raises(ValueError, match="positive reward cap"):
        compute_oracle_regret(0.0, num_keys, total_chests)


def test_metric_split_handles_empty_partition():
    split = summarize_heist_metric_split([0.0, 0.5], [False, False])

    assert split["overall"]["mean"] == pytest.approx(0.25)
    assert split["id"]["count"] == 2
    assert split["ood"] == {
        "count": 0,
        "mean": None,
        "std": None,
        "median": None,
        "min": None,
        "max": None,
    }


def test_metric_split_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="equal lengths"):
        summarize_heist_metric_split([0.0], [])


def test_timeout_fraction_handles_empty_and_mixed_episodes():
    assert timeout_fraction([]) is None
    assert timeout_fraction([True, False, False, True]) == 0.5


def test_build_heist_metric_summary_preserves_flat_result_schema():
    episode_data = new_heist_episode_data()
    append_heist_episode_data(episode_data, terminal_info())
    append_heist_episode_data(
        episode_data,
        terminal_info(
            **{
                "prev_level/keys_collected": 5,
                "prev_level/num_keys": 4,
                "prev_level/total_chests": 2,
                "prev_level/chests_opened": 1,
                "prev_level_complete": 0,
            }
        ),
    )

    result = build_heist_metric_summary(
        env_returns=[3.0, 1.0], episode_data=episode_data, level_ood_gt=[False, True]
    ).to_result_dict()

    assert result["oracle_regret"] == [0.0, 0.5]
    assert result["surplus_keys"] == [1.0, 4.0]
    assert result["mean_oracle_regret"] == pytest.approx(0.25)
    assert result["id_mean_oracle_regret"] == 0.0
    assert result["ood_mean_oracle_regret"] == 0.5
    assert result["mean_surplus_keys"] == pytest.approx(2.5)
    assert result["timeout_fraction"] == 0.5
    assert result["id_timeout_fraction"] == 0.0
    assert result["ood_timeout_fraction"] == 1.0
    assert result["keys_collected"] == [4, 5]
    assert result["level_complete"] == [True, False]


def test_build_heist_metric_summary_rejects_misaligned_counter_array():
    episode_data = new_heist_episode_data()
    append_heist_episode_data(episode_data, terminal_info())
    episode_data["total_steps"].clear()

    with pytest.raises(ValueError, match="total_steps has 0 values for 1 returns"):
        build_heist_metric_summary([3.0], episode_data, [False])


@pytest.mark.parametrize("random_percent", [0, 100])
def test_real_heist_env_exposes_terminal_episode_counters(random_percent):
    env = ProcgenEnv(
        num_envs=1,
        env_name="heist_afh",
        num_levels=1,
        start_level=123,
        distribution_mode="hard",
        random_percent=random_percent,
        timeout=1,
    )
    env.reset()

    try:
        _, _, done, info = env.step(np.array([0], dtype=np.int32))
    finally:
        env.close()

    assert done[0]
    episode = extract_heist_episode_data(info[0])
    assert episode["num_keys"] > 0
    assert episode["total_chests"] > 0
    assert episode["total_steps"] == 1
    assert not episode["level_complete"]
    if random_percent == 0:
        assert episode["total_chests"] == episode["num_keys"] * 2
        assert info[0]["prev_level/randomize_goal"] == 0
    else:
        assert episode["num_keys"] == episode["total_chests"] * 2
        assert info[0]["prev_level/randomize_goal"] == 1

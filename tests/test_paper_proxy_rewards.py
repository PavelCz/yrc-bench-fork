import numpy as np

from analyzing.paper_proxy_rewards import paper_episode_returns
from analyzing.utils import extract_from_data


def test_coinrun_maze_penalty_maps_fail_after_proxy_to_zero():
    summary = {
        "raw_returns": [10.0, -5.0, 5.0, 0.0, 10.0, 0.0],
        "level_ood_gt": [False, True, True, True, True, False],
        "invisible_coin_collected": [True, True, True, False, False, False],
    }
    np.testing.assert_allclose(
        paper_episode_returns(summary),
        [10.0, 0.0, 5.0, 0.0, 10.0, 0.0],
    )


def test_heist_penalty_maps_recorded_minus_five_to_minus_two():
    summary = {
        "raw_returns": [3.0, -2.0, 2.0, 4.0],
        "level_ood_gt": [False, True, True, True],
        "chests_opened": [3, 3, 2, 4],
        "all_keys_triggered": [True, True, False, True],
    }
    np.testing.assert_allclose(
        paper_episode_returns(summary),
        [3.0, 1.0, 2.0, 4.0],
    )


def test_heist_penalty_with_chests_at_proxy_keeps_recorded_returns():
    summary = {
        "raw_returns": [3.0, 2.5],
        "level_ood_gt": [False, True],
        "chests_opened": [3, 3],
        "all_keys_triggered": [True, True],
        "chests_at_proxy": [3, 2],
    }
    np.testing.assert_allclose(paper_episode_returns(summary), [3.0, 2.5])


def test_heist_fail_recording_is_unchanged():
    summary = {
        "raw_returns": [3.0, 2.0],
        "level_ood_gt": [True, True],
        "chests_opened": [3, 2],
        "all_keys_triggered": [True, True],
    }
    np.testing.assert_allclose(paper_episode_returns(summary), [3.0, 2.0])


def test_extract_from_data_performance_uses_paper_returns():
    test_summary = {
        "raw_returns": [10.0, -5.0],
        "level_ood_gt": [True, True],
        "level_ood_pred": [False, False],
        "invisible_coin_collected": [False, True],
    }
    data = {
        "meta": np.array([{"summary": {"test": test_summary}}], dtype=object),
        "performances": np.array([2.5]),
        "order": np.array([1]),
    }
    np.testing.assert_allclose(extract_from_data(data, "performance"), [5.0])
    np.testing.assert_allclose(extract_from_data(data, "ood_performance"), [5.0])

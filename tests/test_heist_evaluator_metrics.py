from types import SimpleNamespace

import numpy as np
import pytest

from YRC.core.evaluator import Evaluator, _heist_wandb_metrics


def make_config(tmp_path, env_name):
    return SimpleNamespace(
        evaluation=SimpleNamespace(
            defer_to_oracle=False,
            act_greedy=True,
            video_filter=["all"],
            video_filter_mode="any",
            video_episodes_to_collect=1,
        ),
        eval_run_dir=tmp_path,
        coord_policy=SimpleNamespace(metric="max_prob"),
        algorithm=SimpleNamespace(cls=None),
        environment=SimpleNamespace(
            common=SimpleNamespace(env_name=env_name),
            test=SimpleNamespace(num_levels=2),
        ),
    )


class FakePolicy:
    def eval(self):
        return self

    def act(self, obs, greedy, return_scores_and_recons):
        del obs, greedy, return_scores_and_recons
        return np.array([0]), None, None

    def reset_rolling_average_buffer(self, env_idx):
        del env_idx


class FakeHeistEnv:
    num_envs = 1

    def __init__(self, terminal_infos):
        self.terminal_infos = terminal_infos
        self.step_index = 0
        self.closed = False

    def reset(self):
        return {"env_obs": np.zeros((1, 3, 2, 2), dtype=np.float32)}

    def step(self, action):
        del action
        info = dict(self.terminal_infos[self.step_index % len(self.terminal_infos)])
        self.step_index += 1
        obs = {"env_obs": np.zeros((1, 3, 2, 2), dtype=np.float32)}
        # Deliberately make shaped reward differ from raw env_reward.
        reward = np.array([99.0], dtype=np.float32)
        done = np.array([True])
        return obs, reward, done, [info]

    def close(self):
        self.closed = True


def terminal_info(
    *,
    env_reward,
    is_ood,
    level_seed,
    keys_collected,
    num_keys,
    total_chests,
    chests_opened,
    level_complete,
):
    return {
        "env_reward": env_reward,
        "randomize_goal": int(is_ood),
        "prev_level/randomize_goal": int(is_ood),
        "prev_level_seed": level_seed,
        "prev_level/keys_collected": keys_collected,
        "prev_level/num_keys": num_keys,
        "prev_level/total_chests": total_chests,
        "prev_level/chests_opened": chests_opened,
        "prev_level/total_steps": 12,
        "prev_level_complete": int(level_complete),
    }


def test_evaluator_collects_heist_metrics_using_raw_env_reward(tmp_path):
    infos = [
        terminal_info(
            env_reward=2.0,
            is_ood=False,
            level_seed=10,
            keys_collected=2,
            num_keys=2,
            total_chests=4,
            chests_opened=2,
            level_complete=True,
        ),
        terminal_info(
            env_reward=1.0,
            is_ood=True,
            level_seed=11,
            keys_collected=4,
            num_keys=4,
            total_chests=2,
            chests_opened=1,
            level_complete=False,
        ),
    ]
    env = FakeHeistEnv(infos)
    config = make_config(tmp_path, "heist_afh")
    evaluator = Evaluator(config, config.environment)

    summary = evaluator.eval(FakePolicy(), {"test": env}, ["test"], num_episodes=2)[
        "test"
    ]

    assert env.closed
    assert summary["raw_returns"] == [99.0, 99.0]
    assert summary["env_return_mean"] == 1.5
    assert summary["level_seeds"] == [10, 11]
    assert summary["level_ood_gt"] == [False, True]
    assert summary["oracle_regret"] == [0.0, 0.5]
    assert summary["surplus_keys"] == [0.0, 3.0]
    assert summary["redundant_key_triggered"] == [False, True]
    assert summary["all_keys_triggered"] == [True, True]
    assert summary["keys_collected"] == [2, 4]
    assert summary["level_complete"] == [True, False]
    assert summary["mean_oracle_regret"] == pytest.approx(0.25)
    assert summary["id_mean_oracle_regret"] == 0.0
    assert summary["ood_mean_oracle_regret"] == 0.5
    assert summary["redundant_key_trigger_rate"] == 0.5
    assert summary["id_redundant_key_trigger_rate"] == 0.0
    assert summary["ood_redundant_key_trigger_rate"] == 1.0
    assert summary["all_keys_trigger_rate"] == 1.0
    assert summary["id_all_keys_trigger_rate"] == 1.0
    assert summary["ood_all_keys_trigger_rate"] == 1.0
    assert summary["timeout_fraction"] == 0.5
    assert summary["id_timeout_fraction"] == 0.0
    assert summary["ood_timeout_fraction"] == 1.0


@pytest.mark.parametrize(
    ("logging_mode", "expected_calls"),
    [("folder", 1), ("none", 0)],
)
def test_folder_videos_are_processed_without_wandb_logger(
    tmp_path, monkeypatch, logging_mode, expected_calls
):
    info = terminal_info(
        env_reward=2.0,
        is_ood=False,
        level_seed=10,
        keys_collected=2,
        num_keys=2,
        total_chests=4,
        chests_opened=2,
        level_complete=True,
    )
    config = make_config(tmp_path, "heist_afh")
    config.evaluation.video_logging_mode = logging_mode
    evaluator = Evaluator(config, config.environment)
    process_calls = []
    monkeypatch.setattr(
        evaluator,
        "_process_and_log_videos",
        lambda *args: process_calls.append(args),
    )

    evaluator.eval(
        FakePolicy(),
        {"test": FakeHeistEnv([info])},
        ["test"],
        num_episodes=1,
        logger=None,
    )

    assert len(process_calls) == expected_calls


def test_evaluator_rejects_incomplete_heist_terminal_info(tmp_path):
    info = terminal_info(
        env_reward=1.0,
        is_ood=True,
        level_seed=11,
        keys_collected=4,
        num_keys=4,
        total_chests=2,
        chests_opened=1,
        level_complete=False,
    )
    del info["prev_level/chests_opened"]
    config = make_config(tmp_path, "heist_afh")
    evaluator = Evaluator(config, config.environment)

    with pytest.raises(KeyError, match="prev_level/chests_opened"):
        evaluator.eval(
            FakePolicy(), {"test": FakeHeistEnv([info])}, ["test"], num_episodes=1
        )


def test_evaluator_handles_empty_ood_metric_split(tmp_path):
    info = terminal_info(
        env_reward=2.0,
        is_ood=False,
        level_seed=10,
        keys_collected=2,
        num_keys=2,
        total_chests=4,
        chests_opened=2,
        level_complete=True,
    )
    config = make_config(tmp_path, "heist_afh")
    evaluator = Evaluator(config, config.environment)

    summary = evaluator.eval(
        FakePolicy(), {"test": FakeHeistEnv([info])}, ["test"], num_episodes=1
    )["test"]

    assert summary["ood_mean_oracle_regret"] is None
    assert summary["ood_mean_surplus_keys"] is None
    assert summary["ood_redundant_key_trigger_rate"] is None
    assert summary["ood_all_keys_trigger_rate"] is None
    assert summary["ood_timeout_fraction"] is None


def test_non_heist_summary_does_not_gain_heist_fields(tmp_path):
    config = make_config(tmp_path, "coinrun")
    evaluator = Evaluator(config, config.environment)
    log = {
        "returns": [1.0],
        "env_returns": [1.0],
        "episode_length": [2],
        "action_1": [0],
        "level_ood_pred": [False],
        "level_ood_gt": [False],
        "num_finished_episodes": 1,
        "invisible_coin_collected": [False],
        "first_ood_timestep": [None],
        "level_seeds": [5],
    }

    summary = evaluator.summarize(log)

    assert "oracle_regret" not in summary
    assert "redundant_key_trigger_rate" not in summary
    assert "all_keys_trigger_rate" not in summary
    assert "timeout_fraction" not in summary


def test_heist_wandb_metrics_include_only_available_scalars():
    metrics = _heist_wandb_metrics(
        {
            "mean_oracle_regret": 0.25,
            "id_mean_oracle_regret": 0.0,
            "ood_mean_oracle_regret": 0.5,
            "mean_surplus_keys": None,
            "redundant_key_trigger_rate": 0.4,
            "id_redundant_key_trigger_rate": 0.1,
            "ood_redundant_key_trigger_rate": 0.7,
            "all_keys_trigger_rate": None,
        }
    )

    assert metrics == {
        "heist/mean_oracle_regret": 0.25,
        "heist/id_mean_oracle_regret": 0.0,
        "heist/ood_mean_oracle_regret": 0.5,
        "heist/redundant_key_trigger_rate": 0.4,
        "heist/id_redundant_key_trigger_rate": 0.1,
        "heist/ood_redundant_key_trigger_rate": 0.7,
    }

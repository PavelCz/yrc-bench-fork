from types import SimpleNamespace

import numpy as np

from YRC.core.evaluator import Evaluator


def make_config(tmp_path):
    return SimpleNamespace(
        evaluation=SimpleNamespace(
            defer_to_oracle=False,
            act_greedy=True,
            video_filter=["all"],
            video_filter_mode="any",
            video_episodes_to_collect=1,
            video_logging_mode="none",
        ),
        eval_run_dir=tmp_path,
        coord_policy=SimpleNamespace(metric="max_logit"),
        algorithm=SimpleNamespace(cls=None),
        environment=SimpleNamespace(
            common=SimpleNamespace(env_name="coinrun"),
            test=SimpleNamespace(num_levels=2),
        ),
    )


class FakePolicy:
    def __init__(self, scores_by_step):
        self.scores_by_step = scores_by_step
        self.step_index = 0

    def eval(self):
        return self

    def act(self, obs, greedy, return_scores_and_recons):
        del obs, greedy, return_scores_and_recons
        scores = self.scores_by_step[self.step_index]
        self.step_index += 1
        if scores is None:
            batch_size = 1
        else:
            batch_size = len(scores)
        return np.zeros(batch_size, dtype=np.int64), scores, None

    def reset_rolling_average_buffer(self, env_idx):
        del env_idx


class FakeVectorEnv:
    def __init__(self, done_by_step, exhausted_by_step=None):
        self.done_by_step = [np.asarray(done, dtype=bool) for done in done_by_step]
        self.num_envs = len(self.done_by_step[0])
        if exhausted_by_step is None:
            exhausted_by_step = [None] * len(self.done_by_step)
        self.exhausted_by_step = exhausted_by_step
        self.step_index = 0
        self.closed = False

    def reset(self):
        return {"env_obs": np.zeros((self.num_envs, 3, 2, 2), dtype=np.float32)}

    def step(self, action):
        del action
        done = self.done_by_step[self.step_index]
        exhausted = self.exhausted_by_step[self.step_index]
        self.step_index += 1
        obs = {"env_obs": np.zeros((self.num_envs, 3, 2, 2), dtype=np.float32)}
        reward = np.ones(self.num_envs, dtype=np.float32)
        infos = [
            {
                "randomize_goal": 0,
                "prev_level/randomize_goal": 0,
                "prev_level_seed": self.step_index * 10 + worker_idx,
                "seeds_exhausted": bool(exhausted[worker_idx])
                if exhausted is not None
                else False,
            }
            for worker_idx in range(self.num_envs)
        ]
        return obs, reward, done, infos

    def close(self):
        self.closed = True


def test_episode_max_scores_include_terminal_action_and_reset_per_worker(tmp_path):
    config = make_config(tmp_path)
    evaluator = Evaluator(config, config.environment)
    env = FakeVectorEnv(
        [
            [False, True],
            [True, False],
            [True, True],
        ]
    )
    policy = FakePolicy(
        [
            np.asarray([12.0, 8.0]),
            np.asarray([9.0, 4.0]),
            np.asarray([2.0, 10.0]),
        ]
    )

    summary = evaluator.eval(policy, {"test": env}, ["test"], num_episodes=3)["test"]

    # Worker 0's first episode keeps its earlier maximum of 12 even though its
    # terminal action score is 9. The next episode starts fresh at score 2.
    assert summary["episode_max_scores"] == [8.0, 12.0, 2.0]
    assert len(summary["episode_max_scores"]) == len(summary["raw_returns"]) == 3
    assert len(summary["episode_max_scores"]) == len(summary["episode_lengths"])
    # Both workers finish on the final vector step, but only one episode fits
    # under the requested retention cap.
    assert summary["num_finished_episodes"] == 4
    assert env.closed


def test_scoreless_policy_records_unavailable_value_for_each_episode(tmp_path):
    config = make_config(tmp_path)
    evaluator = Evaluator(config, config.environment)
    env = FakeVectorEnv([[True], [True]])
    policy = FakePolicy([None, None])

    summary = evaluator.eval(policy, {"test": env}, ["test"], num_episodes=2)["test"]

    assert len(summary["episode_max_scores"]) == 2
    assert np.isnan(summary["episode_max_scores"]).all()


def test_nonfinite_or_missing_action_score_invalidates_episode_max(tmp_path):
    config = make_config(tmp_path)
    evaluator = Evaluator(config, config.environment)
    env = FakeVectorEnv([[False], [False], [True]])
    policy = FakePolicy([np.asarray([1.0]), np.asarray([np.nan]), np.asarray([3.0])])

    summary = evaluator.eval(policy, {"test": env}, ["test"], num_episodes=1)["test"]

    assert len(summary["episode_max_scores"]) == 1
    assert np.isnan(summary["episode_max_scores"][0])


def test_rolling_buffer_negative_infinity_warmup_can_be_followed_by_finite_max(
    tmp_path,
):
    config = make_config(tmp_path)
    evaluator = Evaluator(config, config.environment)
    env = FakeVectorEnv([[False], [True]])
    policy = FakePolicy([np.asarray([-np.inf]), np.asarray([2.5])])

    summary = evaluator.eval(policy, {"test": env}, ["test"], num_episodes=1)["test"]

    assert summary["episode_max_scores"] == [2.5]


def test_partially_missing_action_score_invalidates_episode_max(tmp_path):
    config = make_config(tmp_path)
    evaluator = Evaluator(config, config.environment)
    env = FakeVectorEnv([[False], [False], [True]])
    policy = FakePolicy([np.asarray([1.0]), None, np.asarray([3.0])])

    summary = evaluator.eval(policy, {"test": env}, ["test"], num_episodes=1)["test"]

    assert len(summary["episode_max_scores"]) == 1
    assert np.isnan(summary["episode_max_scores"][0])


def test_worker_reporting_exhausted_seeds_does_not_add_later_scores(tmp_path):
    config = make_config(tmp_path)
    evaluator = Evaluator(config, config.environment)
    env = FakeVectorEnv(
        [[True, False], [True, False], [False, True]],
        exhausted_by_step=[[True, False], [True, False], [True, False]],
    )
    policy = FakePolicy(
        [
            np.asarray([5.0, 1.0]),
            np.asarray([100.0, 2.0]),
            np.asarray([200.0, 3.0]),
        ]
    )

    summary = evaluator.eval(policy, {"test": env}, ["test"], num_episodes=3)["test"]

    # The 5.0 score belongs to the step that exhausted worker 0 and is kept.
    # A spurious later terminal pulse from that worker is retained by the
    # existing episode accounting, but carries no score evidence.
    assert summary["episode_max_scores"][0] == 5.0
    assert np.isnan(summary["episode_max_scores"][1])
    assert summary["episode_max_scores"][2] == 3.0

import subprocess
import sys

import numpy as np
import pytest

from procgen import ProcgenEnv, ProcgenGym3Env
from YRC.core.evaluator import (
    _episode_randomize_goal,
    _step_invisible_coin_collected,
)


def test_terminal_procgen_info_uses_prev_level_fields():
    info = {
        "randomize_goal": 0,
        "prev_level/randomize_goal": 1,
        "invisible_coin_collected": 0,
        "prev_level/invisible_coin_collected": 1,
    }

    assert _episode_randomize_goal(info, done=True, current_value=False)
    assert _step_invisible_coin_collected(info, done=True)


def test_nonterminal_procgen_info_uses_current_fields():
    info = {
        "randomize_goal": 1,
        "prev_level/randomize_goal": 0,
        "invisible_coin_collected": 1,
        "prev_level/invisible_coin_collected": 0,
    }

    assert _episode_randomize_goal(info, done=False, current_value=True)
    assert _step_invisible_coin_collected(info, done=False)


def test_coinrun_proxy_fail_terminates_on_randomized_proxy_coin():
    env = ProcgenEnv(
        num_envs=1,
        env_name="coinrun_proxy_fail",
        num_levels=1,
        start_level=0,
        distribution_mode="hard",
        random_percent=100,
    )
    env.reset()

    reward = None
    done = None
    info = None
    terminal_step = None
    action_pattern = [8, 7, 7, 7, 7]

    try:
        for step_idx in range(200):
            action = np.array(
                [action_pattern[step_idx % len(action_pattern)]], dtype=np.int32
            )
            _, reward, done, info = env.step(action)
            if done[0]:
                terminal_step = step_idx + 1
                break
    finally:
        env.close()

    assert terminal_step is not None
    assert terminal_step == 87
    assert reward[0] == 0
    assert done[0]
    assert info[0]["prev_level/invisible_coin_collected"] == 1
    assert info[0]["prev_level/randomize_goal"] == 1


def test_coinrun_proxy_penalty_applies_once_without_terminating():
    env = ProcgenEnv(
        num_envs=1,
        env_name="coinrun_proxy_penalty",
        num_levels=1,
        start_level=0,
        distribution_mode="hard",
        random_percent=100,
    )
    env.reset()

    penalty_steps = []
    action_pattern = [8, 7, 7, 7, 7]

    try:
        for step_idx in range(110):
            action = np.array(
                [action_pattern[step_idx % len(action_pattern)]], dtype=np.int32
            )
            _, reward, done, info = env.step(action)
            if reward[0] == -5:
                penalty_steps.append(step_idx + 1)
                assert not done[0]
                assert info[0]["randomize_goal"] == 1
                assert info[0]["invisible_coin_collected"] == 1
            elif info[0]["invisible_coin_collected"] == 1:
                assert reward[0] >= 0
    finally:
        env.close()

    assert penalty_steps == [87]


def test_coinrun_proxy_penalty_survives_state_restore():
    env_kwargs = {
        "num": 1,
        "env_name": "coinrun_proxy_penalty",
        "num_levels": 1,
        "start_level": 0,
        "distribution_mode": "hard",
        "random_percent": 100,
    }
    source_env = ProcgenGym3Env(**env_kwargs)
    restored_env = ProcgenGym3Env(**env_kwargs)
    action_pattern = [8, 7, 7, 7, 7]
    state = None

    try:
        for step_idx in range(110):
            source_env.act(
                np.array(
                    [action_pattern[step_idx % len(action_pattern)]], dtype=np.int32
                )
            )
            reward, _, _ = source_env.observe()
            if reward[0] == -5:
                state = source_env.callmethod("get_state")
                break

        assert state is not None
        restored_env.callmethod("set_state", state)
        restored_env.act(np.array([4], dtype=np.int32))
        restored_reward, _, _ = restored_env.observe()
        restored_info = restored_env.get_info()[0]
    finally:
        source_env.close()
        restored_env.close()

    assert restored_reward[0] == 0
    assert restored_info["invisible_coin_collected"] == 1


def test_maze_proxy_fail_triggers_in_ood_levels():
    env = ProcgenEnv(
        num_envs=4,
        env_name="maze_proxy_fail",
        num_levels=200,
        start_level=0,
        distribution_mode="hard",
        random_percent=100,
    )
    env.reset()
    rng = np.random.default_rng(0)

    proxy_terminations = 0
    goal_terminations = 0
    total_terminations = 0
    proxy_reward_violations = 0
    proxy_without_randomize_flag = 0

    try:
        for _ in range(20000):
            action = rng.integers(0, 15, size=4, dtype=np.int32)
            _, reward, done, info = env.step(action)
            for i in range(4):
                if not done[i]:
                    continue
                total_terminations += 1
                if info[i]["prev_level/invisible_coin_collected"] == 1:
                    proxy_terminations += 1
                    if reward[i] != 0:
                        proxy_reward_violations += 1
                    if info[i]["prev_level/randomize_goal"] != 1:
                        proxy_without_randomize_flag += 1
                elif reward[i] > 0:
                    goal_terminations += 1
            if total_terminations >= 100:
                break
    finally:
        env.close()

    assert proxy_terminations > 0, "expected at least one proxy termination"
    assert proxy_reward_violations == 0
    assert proxy_without_randomize_flag == 0


def test_maze_proxy_penalty_applies_once_and_allows_later_success():
    env = ProcgenEnv(
        num_envs=4,
        env_name="maze_proxy_penalty",
        num_levels=200,
        start_level=0,
        distribution_mode="hard",
        random_percent=100,
        rand_seed=8,
    )
    env.reset()
    rng = np.random.default_rng(0)
    episode_rewards = np.zeros(4)
    penalty_seen = np.zeros(4, dtype=bool)

    penalty_events = 0
    repeated_penalties = 0
    successes_after_penalty = 0

    try:
        for _ in range(10000):
            action = rng.integers(0, 15, size=4, dtype=np.int32)
            _, reward, done, info = env.step(action)
            episode_rewards += reward

            for i in range(4):
                if reward[i] == -5:
                    repeated_penalties += int(penalty_seen[i])
                    penalty_seen[i] = True
                    penalty_events += 1
                    assert not done[i]
                    assert info[i]["randomize_goal"] == 1
                    assert info[i]["invisible_coin_collected"] == 1
                elif reward[i] < 0:
                    repeated_penalties += 1

                if done[i]:
                    if penalty_seen[i] and episode_rewards[i] == 5:
                        successes_after_penalty += 1
                    episode_rewards[i] = 0
                    penalty_seen[i] = False

            if penalty_events >= 3 and successes_after_penalty > 0:
                break
    finally:
        env.close()

    assert penalty_events >= 3
    assert repeated_penalties == 0
    assert successes_after_penalty > 0


def test_maze_proxy_penalty_survives_state_restore():
    env_kwargs = {
        "num": 4,
        "env_name": "maze_proxy_penalty",
        "num_levels": 200,
        "start_level": 0,
        "distribution_mode": "hard",
        "random_percent": 100,
        "rand_seed": 8,
    }
    source_env = ProcgenGym3Env(**env_kwargs)
    restored_env = ProcgenGym3Env(**env_kwargs)
    rng = np.random.default_rng(0)
    state = None
    penalized_env = None

    try:
        for _ in range(10000):
            source_env.act(rng.integers(0, 15, size=4, dtype=np.int32))
            reward, _, _ = source_env.observe()
            penalized = np.flatnonzero(reward == -5)
            if len(penalized) > 0:
                penalized_env = int(penalized[0])
                state = source_env.callmethod("get_state")
                break

        assert state is not None
        assert penalized_env is not None
        restored_env.callmethod("set_state", state)
        restored_env.act(np.full(4, 4, dtype=np.int32))
        restored_reward, _, _ = restored_env.observe()
        restored_info = restored_env.get_info()[penalized_env]
    finally:
        source_env.close()
        restored_env.close()

    assert restored_reward[penalized_env] == 0
    assert restored_info["invisible_coin_collected"] == 1


@pytest.mark.parametrize("env_name", ["coinrun_proxy_penalty", "maze_proxy_penalty"])
def test_proxy_penalty_inactive_on_id_levels(env_name):
    num_envs = 4 if env_name.startswith("maze") else 1
    env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=200,
        start_level=0,
        distribution_mode="hard",
        random_percent=0,
    )
    env.reset()
    rng = np.random.default_rng(0)

    negative_rewards = 0
    successful_episodes = 0

    try:
        for step_idx in range(2000):
            if env_name.startswith("maze"):
                action = rng.integers(0, 15, size=num_envs, dtype=np.int32)
            else:
                action_pattern = [8, 7, 7, 7, 7]
                action = np.array(
                    [action_pattern[step_idx % len(action_pattern)]], dtype=np.int32
                )

            _, reward, done, _ = env.step(action)
            negative_rewards += int(np.count_nonzero(reward < 0))
            successful_episodes += sum(
                bool(done[i] and reward[i] == 10) for i in range(num_envs)
            )
            if successful_episodes > 0:
                break
    finally:
        env.close()

    assert negative_rewards == 0
    assert successful_episodes > 0


def test_maze_proxy_fail_inactive_when_random_percent_zero():
    env = ProcgenEnv(
        num_envs=4,
        env_name="maze_proxy_fail",
        num_levels=200,
        start_level=0,
        distribution_mode="hard",
        random_percent=0,
    )
    env.reset()
    rng = np.random.default_rng(0)

    proxy_terminations = 0
    total_terminations = 0

    try:
        for _ in range(10000):
            action = rng.integers(0, 15, size=4, dtype=np.int32)
            _, _, done, info = env.step(action)
            for i in range(4):
                if not done[i]:
                    continue
                total_terminations += 1
                if info[i]["prev_level/invisible_coin_collected"] == 1:
                    proxy_terminations += 1
            if total_terminations >= 50:
                break
    finally:
        env.close()

    assert total_terminations >= 50
    assert proxy_terminations == 0


@pytest.mark.parametrize("env_name", ["maze_proxy_fail", "maze_proxy_penalty"])
def test_maze_proxy_variants_reject_nonzero_rand_region(env_name):
    # `fatal()` calls exit() in-process, so run in a subprocess to capture it.
    code = (
        "from procgen import ProcgenEnv\n"
        f"env = ProcgenEnv(num_envs=1, env_name='{env_name}', num_levels=1,\n"
        "                 start_level=0, distribution_mode='hard',\n"
        "                 random_percent=100, rand_region=3)\n"
        "env.reset()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert f"{env_name} requires rand_region=0" in combined
    assert "rand_region=3" in combined

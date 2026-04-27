import subprocess
import sys

import numpy as np

from procgen import ProcgenEnv
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


def test_maze_proxy_fail_rejects_nonzero_rand_region():
    # `fatal()` calls exit() in-process, so run in a subprocess to capture it.
    code = (
        "from procgen import ProcgenEnv\n"
        "env = ProcgenEnv(num_envs=1, env_name='maze_proxy_fail', num_levels=1,\n"
        "                 start_level=0, distribution_mode='hard',\n"
        "                 random_percent=100, rand_region=3)\n"
        "env.reset()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "maze_proxy_fail requires rand_region=0" in combined
    assert "rand_region=3" in combined

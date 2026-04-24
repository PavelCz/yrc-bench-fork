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

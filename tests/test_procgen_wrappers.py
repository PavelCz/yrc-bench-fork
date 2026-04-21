import gym
import numpy as np

from YRC.envs.procgen.wrappers import HardResetWrapper, VecEnv


class DummyVecEnv(VecEnv):
    def __init__(self):
        super().__init__(
            num_envs=2,
            observation_space=gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Discrete(3),
        )
        self.reset_calls = 0
        self.actions = []

    def reset(self):
        self.reset_calls += 1
        return np.array([[1.0], [2.0]], dtype=np.float32)

    def step_async(self, actions):
        self.actions.append(np.array(actions))

    def step_wait(self):
        obs = np.array([[3.0], [4.0]], dtype=np.float32)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.array([False, False])
        info = [{}, {}]
        return obs, reward, done, info


def test_hard_reset_wrapper_defaults_to_forced_reset():
    env = DummyVecEnv()
    wrapped = HardResetWrapper(env)

    obs = wrapped.reset()

    assert env.reset_calls == 0
    assert len(env.actions) == 1
    np.testing.assert_array_equal(env.actions[0], np.array([-1, -1]))
    np.testing.assert_array_equal(obs, np.array([[3.0], [4.0]], dtype=np.float32))


def test_hard_reset_wrapper_can_use_underlying_reset():
    env = DummyVecEnv()
    wrapped = HardResetWrapper(env, force_on_reset=False)

    obs = wrapped.reset()

    assert env.reset_calls == 1
    assert env.actions == []
    np.testing.assert_array_equal(obs, np.array([[1.0], [2.0]], dtype=np.float32))

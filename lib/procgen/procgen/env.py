import os
import random
from typing import Sequence, Optional, List

import gym3
from gym3.libenv import CEnv
import numpy as np
from .build import build

try:
    from imagecorruptions import corrupt
except ImportError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_STATE_SIZE = 2**20

ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "coinrun_mod_wall",
    "coinrun_aisc",
    "coinrun_proxy_fail",
    "dodgeball",
    "fruitbot",
    "heist",
    "heist_afh",
    "heist_aisc_many_chests",
    "heist_aisc_many_keys",
    "jumper",
    "leaper",
    "maze",
    "maze_afh",
    "maze_fixed_size",
    "maze_aisc",
    "maze_proxy_fail",
    "maze_yellowline",
    "maze_redline_yellowgem",
    "maze_yellowstar_redgem",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

EXPLORATION_LEVEL_SEEDS = {
    "coinrun": 1949448038,
    "coinrun_mod_wall": 1949448038,
    "coinrun_aisc": 1949448038,
    "coinrun_proxy_fail": 1949448038,
    "caveflyer": 1259048185,
    "leaper": 1318677581,
    "jumper": 1434825276,
    "maze": 158988835,
    "maze_afh": 158988835,
    "maze_fixed_size": 158988835,
    "maze_aisc": 158988835,
    "maze_proxy_fail": 158988835,
    "maze_yellowline": 158988835,
    "maze_redline_yellowgem": 158988835,
    "maze_yellowstar_redgem": 158988835,
    "heist": 876640971,
    "heist_afh": 876640971,
    "heist_aisc_many_chests": 876640971,
    "heist_aisc_many_keys": 876640971,
    "climber": 1561126160,
    "ninja": 1123500215,
}

# should match DistributionMode in game.h, except for 'exploration' which is handled by Python
DISTRIBUTION_MODE_DICT = {
    "easy": 0,
    "hard": 1,
    "extreme": 2,
    "memory": 10,
    "exploration": 20,
}


def create_random_seed():
    rand_seed = random.SystemRandom().randint(0, 2**31 - 1)
    try:
        # force MPI processes to definitely choose different random seeds
        from mpi4py import MPI

        rand_seed = rand_seed - (rand_seed % MPI.COMM_WORLD.size) + MPI.COMM_WORLD.rank
    except ModuleNotFoundError:
        pass
    return rand_seed


class BaseProcgenEnv(CEnv):
    """
    Base procedurally generated environment
    """

    def __init__(
        self,
        num,
        env_name,
        options,  # ranrom_percent and other extra env options can go through here
        debug=False,
        rand_seed=None,
        num_levels=0,
        start_level=0,
        use_sequential_levels=False,
        debug_mode=0,
        resource_root=None,
        num_threads=4,
        render_mode=None,
        level_seeds: Optional[List[int]] = None,
        level_seeds_mode: str = "sequential",
    ):
        if resource_root is None:
            resource_root = os.path.join(SCRIPT_DIR, "data", "assets") + os.sep
            assert os.path.exists(resource_root)

        lib_dir = os.path.join(SCRIPT_DIR, "data", "prebuilt")
        if os.path.exists(lib_dir):
            assert any(
                [
                    os.path.exists(os.path.join(lib_dir, name))
                    for name in ["libenv.so", "libenv.dylib", "env.dll"]
                ]
            ), "package is installed, but the prebuilt environment library is missing"
            assert not debug, "debug has no effect for pre-compiled library"
        else:
            # only compile if we don't find a pre-built binary
            lib_dir = build(debug=debug)

        self.combos = self.get_combos()

        if render_mode is None:
            render_human = False
        elif render_mode == "rgb_array":
            render_human = True
        else:
            raise Exception(f"invalid render mode {render_mode}")

        if rand_seed is None:
            rand_seed = create_random_seed()

        # Convert level_seeds list to comma-separated string for C++ backend
        level_seeds_str = ""
        seed_container_mode = False
        seed_random_mode = False
        if level_seeds is not None:
            level_seeds_str = ",".join(str(s) for s in level_seeds)
            # Only validate level_seeds_mode when level_seeds is provided
            assert level_seeds_mode in ("sequential", "container", "random"), (
                f'level_seeds_mode must be "sequential", "container", or "random", got "{level_seeds_mode}"'
            )
            seed_container_mode = level_seeds_mode == "container"
            seed_random_mode = level_seeds_mode == "random"

        options.update(
            {
                "env_name": env_name,
                "num_levels": num_levels,
                "start_level": start_level,
                "num_actions": len(self.combos),
                "use_sequential_levels": bool(use_sequential_levels),
                "debug_mode": debug_mode,
                "rand_seed": rand_seed,
                "num_threads": num_threads,
                "render_human": render_human,
                "level_seeds": level_seeds_str,
                "seed_container_mode": seed_container_mode,
                "seed_random_mode": seed_random_mode,
                # these will only be used the first time an environment is created in a process
                "resource_root": resource_root,
            }
        )

        self.options = options

        super().__init__(
            lib_dir=lib_dir,
            num=num,
            options=options,
            c_func_defs=[
                "int get_state(libenv_env *, int, char *, int);",
                "void set_state(libenv_env *, int, char *, int);",
                "void reset_remaining_timeout(libenv_env *, int, int);",
            ],
        )
        # don't use the dict space for actions
        self.ac_space = self.ac_space["action"]

    def get_state(self):
        length = MAX_STATE_SIZE
        buf = self._ffi.new(f"char[{length}]")
        result = []
        for env_idx in range(self.num):
            n = self.call_c_func("get_state", env_idx, buf, length)
            result.append(bytes(self._ffi.buffer(buf, n)))
        return result

    def set_state(self, states):
        assert len(states) == self.num
        for env_idx in range(self.num):
            state = states[env_idx]
            self.call_c_func("set_state", env_idx, state, len(state))

    def reset_remaining_timeout(self, env_idx: int, remaining_steps: int) -> None:
        if env_idx < 0 or env_idx >= self.num:
            raise IndexError(f"env_idx must be in [0, {self.num}), got {env_idx}")
        if remaining_steps <= 0:
            raise ValueError(f"remaining_steps must be positive, got {remaining_steps}")
        self.call_c_func(
            "reset_remaining_timeout",
            int(env_idx),
            int(remaining_steps),
        )

    def get_combos(self):
        return [
            ("LEFT", "DOWN"),
            ("LEFT",),
            ("LEFT", "UP"),
            ("DOWN",),
            (),
            ("UP",),
            ("RIGHT", "DOWN"),
            ("RIGHT",),
            ("RIGHT", "UP"),
            ("D",),
            ("A",),
            ("W",),
            ("S",),
            ("Q",),
            ("E",),
        ]

    def keys_to_act(
        self, keys_list: Sequence[Sequence[str]]
    ) -> List[Optional[np.ndarray]]:
        """
        Convert list of keys being pressed to actions, used in interactive mode
        """
        result = []
        for keys in keys_list:
            action = None
            max_len = -1
            for i, combo in enumerate(self.get_combos()):
                pressed = True
                for key in combo:
                    if key not in keys:
                        pressed = False

                if pressed and (max_len < len(combo)):
                    action = i
                    max_len = len(combo)

            if action is not None:
                action = np.array([action])
            result.append(action)
        return result

    def act(self, ac):
        # tensorflow may return int64 actions (https://github.com/openai/gym/blob/master/gym/spaces/discrete.py#L13)
        # so always cast actions to int32
        return super().act({"action": ac.astype(np.int32)})


class ProcgenGym3Env(BaseProcgenEnv):
    """
    gym3 interface for Procgen
    """

    def __init__(
        self,
        num,
        env_name,
        center_agent=True,
        use_backgrounds=True,
        use_monochrome_assets=False,
        restrict_themes=False,
        use_generated_assets=False,
        paint_vel_info=False,
        distribution_mode="hard",
        random_percent=0,
        key_penalty=0,
        step_penalty=0,
        rand_region=0,
        corruption_type=None,
        corruption_severity=1,
        continue_after_coin=False,
        randomize_agent_start=False,
        timeout=None,
        **kwargs,
    ):
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        assert distribution_mode in DISTRIBUTION_MODE_DICT, (
            f'"{distribution_mode}" is not a valid distribution mode.'
        )

        if distribution_mode == "exploration":
            assert env_name in EXPLORATION_LEVEL_SEEDS, (
                f"{env_name} does not support exploration mode"
            )

            distribution_mode = DISTRIBUTION_MODE_DICT["hard"]
            assert "num_levels" not in kwargs, "exploration mode overrides num_levels"
            kwargs["num_levels"] = 1
            assert "start_level" not in kwargs, "exploration mode overrides start_level"
            kwargs["start_level"] = EXPLORATION_LEVEL_SEEDS[env_name]
        else:
            distribution_mode = DISTRIBUTION_MODE_DICT[distribution_mode]

        options = {
            "center_agent": bool(center_agent),
            "use_generated_assets": bool(use_generated_assets),
            "use_monochrome_assets": bool(use_monochrome_assets),
            "restrict_themes": bool(restrict_themes),
            "use_backgrounds": bool(use_backgrounds),
            "paint_vel_info": bool(paint_vel_info),
            "distribution_mode": distribution_mode,
            "random_percent": int(random_percent),
            "key_penalty": int(key_penalty),
            "step_penalty": int(step_penalty),
            "rand_region": int(rand_region),
            "continue_after_coin": bool(continue_after_coin),
            "randomize_agent_start": bool(randomize_agent_start),
        }
        if timeout is not None:
            options["timeout"] = int(timeout)
        super().__init__(num, env_name, options, **kwargs)

    def observe(self):
        """override!"""
        obs = super().observe()
        if self.corruption_type is not None:
            rgb = obs[1]["rgb"]
            rgb = [
                corrupt(
                    img,
                    severity=self.corruption_severity,
                    corruption_name=self.corruption_type,
                )
                for img in rgb
            ]
            rgb = np.array(rgb)
            obs[1]["rgb"] = rgb
        return obs


class ProcgenBaselinesVecEnv(gym3.ToBaselinesVecEnv):
    def reset_remaining_timeout(self, env_idx: int, remaining_steps: int) -> None:
        self.env.reset_remaining_timeout(env_idx, remaining_steps)


def ProcgenEnv(num_envs, env_name, **kwargs):
    """
    Baselines VecEnv interface for Procgen
    """
    return ProcgenBaselinesVecEnv(
        ProcgenGym3Env(num=num_envs, env_name=env_name, **kwargs)
    )

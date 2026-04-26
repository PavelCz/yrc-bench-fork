import logging
from typing import List, Optional
import torch

from procgen import ProcgenEnv
import YRC.envs.procgen.wrappers as wrappers
from YRC.envs.procgen.models import ProcgenModel
from YRC.envs.procgen.policies import EnsemblePolicy as EnsemblePolicy, ProcgenPolicy
from YRC.core.configs.global_configs import get_global_variable


def create_env(
    name,
    config,
    level_seeds: Optional[List[int]] = None,
    level_seeds_mode: str = "sequential",
    render_mode: Optional[str] = "rgb_array",
):
    common_config = config.common

    # These are the config settigns that might depend on the specific mode, i.e. train,
    # val, test
    specific_config = getattr(config, name)

    # Get max_steps if specified in config
    max_steps = getattr(common_config, "max_steps", None)
    randomize_agent_start = getattr(specific_config, "randomize_agent_start", None)
    if randomize_agent_start is None:
        randomize_agent_start = getattr(common_config, "randomize_agent_start", False)

    # Build kwargs for level seeds if provided
    seed_kwargs = {}
    if level_seeds is not None:
        seed_kwargs["level_seeds"] = level_seeds
        seed_kwargs["level_seeds_mode"] = level_seeds_mode

    env = ProcgenEnv(
        env_name=common_config.env_name,
        num_envs=common_config.num_envs,
        num_threads=common_config.num_threads,
        num_levels=specific_config.num_levels,
        start_level=specific_config.start_level,
        distribution_mode=specific_config.distribution_mode,
        rand_seed=specific_config.seed,
        # AFAICT these should be set in the common config
        use_backgrounds=common_config.use_backgrounds,
        use_monochrome_assets=common_config.use_monochrome_assets,
        restrict_themes=common_config.restrict_themes,
        random_percent=specific_config.random_percent,
        randomize_agent_start=randomize_agent_start,
        # Enable human-resolution rendering for video logging when requested.
        render_mode=render_mode,
        # Set episode timeout (max steps) directly in procgen C++ backend
        timeout=max_steps,
        **seed_kwargs,
    )

    env = wrappers.VecExtractDictObs(env, "rgb")
    if common_config.normalize_rew:
        env = wrappers.VecNormalize(
            env, ob=False
        )  # normalizing returns, but not the img frames
    env = wrappers.TransposeFrame(env)
    env = wrappers.ScaledFloatFrame(env)

    # NOTE: this must be done last. In explicit level-seed mode, a forced reset
    # consumes the initial vectorized seed batch and discards those completions.
    env = wrappers.HardResetWrapper(env, force_on_reset=level_seeds is None)
    env.obs_shape = env.observation_space.shape
    return env


def load_policy(path, env):
    model = ProcgenModel(env)
    device = get_global_variable("device")
    model.to(device)
    model.eval()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Loaded model from {path}")

    policy = ProcgenPolicy(model)
    policy.eval()
    return policy

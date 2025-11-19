from pathlib import Path
import os
import time
from typing import List

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy

from YRC.coverage.coverage_search import create_afhp_threshold_sampler
from YRC.coverage.coverage_search import create_ood_percentage_threshold_sampler

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from acs.types import CurvePoint

# Import RandomEnvSwitchWrapper for multi-env switching
from YRC.envs.procgen.wrappers import RandomEnvSwitchWrapper


def create_raw_env_from_config(env_config, base_config):
    """Create a raw (base) environment from custom configuration.

    Args:
        env_config: Configuration object with gym_name and environment settings
        base_config: Base config object to get common settings from
    """
    benchmark = get_global_variable("benchmark")

    if benchmark == "procgen":
        from procgen import ProcgenEnv
        import YRC.envs.procgen.wrappers as wrappers

        # Get common config settings or use defaults
        common_config = base_config.environment.common

        # Helper function to get attribute with fallback for None values
        def get_config_value(obj, attr, default):
            value = getattr(obj, attr, None)
            return default if value is None else value

        # Create environment with custom settings
        # Use getattr for config objects (not .get() which is for dicts)
        env = ProcgenEnv(
            env_name=get_config_value(env_config, "gym_name", common_config.env_name),
            num_envs=get_config_value(env_config, "num_envs", common_config.num_envs),
            num_threads=get_config_value(
                env_config, "num_threads", common_config.num_threads
            ),
            num_levels=get_config_value(env_config, "num_levels", 0),
            start_level=get_config_value(env_config, "start_level", 0),
            distribution_mode=get_config_value(env_config, "distribution_mode", "easy"),
            rand_seed=get_config_value(env_config, "seed", 0),
            use_backgrounds=get_config_value(
                env_config, "use_backgrounds", common_config.use_backgrounds
            ),
            use_monochrome_assets=get_config_value(
                env_config, "use_monochrome_assets", common_config.use_monochrome_assets
            ),
            restrict_themes=get_config_value(
                env_config, "restrict_themes", common_config.restrict_themes
            ),
            random_percent=get_config_value(env_config, "random_percent", 100),
        )

        # Apply standard wrappers
        env = wrappers.VecExtractDictObs(env, "rgb")
        if common_config.normalize_rew:
            env = wrappers.VecNormalize(env, ob=False)
        env = wrappers.TransposeFrame(env)
        env = wrappers.ScaledFloatFrame(env)

        # Apply time limit wrapper if specified
        if hasattr(common_config, "max_steps") and common_config.max_steps is not None:
            env = wrappers.TimeLimitWrapper(env, common_config.max_steps)

        # Must be done last
        env = wrappers.HardResetWrapper(env)
        env.obs_shape = env.observation_space.shape
        env.name = get_config_value(env_config, "gym_name", common_config.env_name)

        return env
    else:
        raise NotImplementedError(
            f"Random env switching not yet implemented for benchmark: {benchmark}"
        )


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    # Record time for profiling purposes
    start_time = time.time()

    # Check if we should use RandomEnvSwitchWrapper
    use_random_env_switch = (
        hasattr(config.evaluation, "use_random_env_switch")
        and config.evaluation.use_random_env_switch
    )

    if use_random_env_switch:
        # Get configuration for random env switching
        env1_config = config.evaluation.random_env_switch.env1
        env2_config = config.evaluation.random_env_switch.env2
        random_percent = config.evaluation.random_env_switch.random_percent

        # Helper function to get attribute with fallback for None values
        def get_config_value(obj, attr, default):
            value = getattr(obj, attr, None)
            return default if value is None else value

        # Use getattr for config objects (not .get() which is for dicts)
        env1_name = get_config_value(env1_config, "gym_name", "env1")
        env2_name = get_config_value(env2_config, "gym_name", "env2")

        print(
            f"Using RandomEnvSwitchWrapper with {env1_name} and {env2_name} "
            f"(random_percent={random_percent})"
        )

        # Create the two base environments with custom configurations
        base_env1 = create_raw_env_from_config(env1_config, config)
        base_env2 = create_raw_env_from_config(env2_config, config)

        # Wrap them with RandomEnvSwitchWrapper
        wrapped_test_env = RandomEnvSwitchWrapper(base_env1, base_env2, random_percent)

        # Create normal envs for train/val
        envs = env_factory.make(config)

        # Replace the test env with our wrapped version
        envs["test"] = envs["train"].__class__(
            config.coord_env,
            wrapped_test_env,
            envs["train"].weak_agent,
            envs["train"].strong_agent,
        )
        # Copy over costs
        envs["test"].strong_query_cost_per_action = envs[
            "train"
        ].strong_query_cost_per_action
        envs["test"].switch_agent_cost_per_action = envs[
            "train"
        ].switch_agent_cost_per_action
        envs["test"].reset()
    else:
        envs = env_factory.make(config)

    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        # The following algorithms do not need to load a model, because they do not
        # need the training step:
        algorithms = ["timestep_random", "level_based_random", "threshold", "heuristic"]
        if config.general.algorithm not in algorithms:
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    # For threshold policy with metrics that require training score distribution (like max_logit),
    # we need to generate scores before running AFHP evaluation
    from YRC.policies.threshold import ThresholdPolicy
    if isinstance(policy, ThresholdPolicy):
        metric = config.coord_policy.metric
        # max_logit requires the training score distribution for percentile computation
        if metric == "max_logit":
            # Use algorithm.num_rollouts if available, otherwise use a default
            num_rollouts = getattr(config.algorithm, "num_rollouts", 256)
            print(f"Generating {num_rollouts} training scores for threshold policy with {metric} metric...")
            policy.generate_scores(envs["train"], num_rollouts)

    evaluator = Evaluator(config, config.environment)

    coverage_fraction = config.evaluation.coverage_fraction
    threshold_sampler: str = config.evaluation.threshold_sampler

    if coverage_fraction < 0.01:
        raise ValueError("Coverage fraction must be at least 0.01")

    # Initialize wandb logger
    save_dir = Path(str(get_global_variable("experiment_dir")))

    # Prepare wandb init parameters
    wandb_kwargs = {
        "name": config.exp_name,
        "project": config.wandb.project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "job_type": "train",
        "config": config,
    }

    if config.wandb.entity is not None:
        wandb_kwargs["entity"] = config.wandb.entity

    exp = wandb.init(**wandb_kwargs)

    wandb_logger = WandbLogger(
        save_dir=save_dir,
        experiment=exp,
    )

    split = "test"

    # Create the joint-coverage sampler via YRC wrapper (adapts to new abcs API)
    max_total_evals = 200

    if threshold_sampler == "afhp":
        sampler = create_afhp_threshold_sampler(
            policy=policy,
            evaluator=evaluator,
            envs=envs,
            split=split,
            coverage_fraction=coverage_fraction,
            max_total_evals=max_total_evals,
            logger=wandb_logger,
        )
    elif threshold_sampler == "ood_percentage":
        sampler = create_ood_percentage_threshold_sampler(
            policy=policy,
            evaluator=evaluator,
            envs=envs,
            split=split,
            coverage_fraction=coverage_fraction,
            max_total_evals=max_total_evals,
            logger=wandb_logger,
        )
    else:
        raise ValueError(f"Invalid threshold sampler: {threshold_sampler}")

    # Run the sampling
    print(
        f"Running joint coverage sampling with coverage_fraction="
        f"{coverage_fraction:.3f}, budget={max_total_evals}..."
    )
    sampling_result = sampler.run()

    # Report coverage
    print(
        f"Coverage x-gap: {sampling_result.coverage_x_max_gap:.3f}, "
        f"y-gap: {sampling_result.coverage_y_max_gap:.3f}"
    )

    # TODO: Rename
    # The sort metric is called afhp for legacy reasons, sort metric or threshold metric
    # would be more appropriate.
    sorted_points: List[CurvePoint] = sorted(
        sampling_result.points, key=lambda p: p.afhp
    )

    total_evals = sampling_result.total_evals

    # Save result summary to file.
    log_file_path = get_global_variable("log_file")
    if log_file_path is None:
        raise ValueError(
            "Log file path is not set. Could not find path to save results."
        )
    log_file_path = Path(log_file_path)
    results_file_path = log_file_path.with_name(
        log_file_path.name.replace(".log", f"_{split}.npz")
    )
    np.savez(
        results_file_path,
        afhps=np.array([pt.afhp for pt in sorted_points]),
        performances=np.array([pt.performance for pt in sorted_points]),
        desired_percentiles=np.array([pt.desired_percentile for pt in sorted_points]),
        meta=np.array([pt.meta for pt in sorted_points]),
        order=np.array([pt.order for pt in sorted_points]),
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {total_evals}")


if __name__ == "__main__":
    main()

from typing import Any

import numpy as np


def calibrate_percentile_mapping(policy, config, evaluator, envs, make_envs, cal_seeds):
    """Calibrate the policy's train_percentile_step/level methods.

    Uses fixed validation seeds (held out from RL training, same ID distribution)
    so calibration is reproducible across runs and independent of num_rollouts.
    """
    from YRC.policies.base import TimestepRandomPolicy
    from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy
    from YRC.policies.ood import OODPolicy
    from YRC.policies.threshold import ThresholdPolicy

    if cal_seeds is None or "cal" not in envs:
        raise ValueError(
            "Calibration requires validation seeds. "
            "Ensure the seed file contains a non-empty 'validation' set and "
            "--level_seeds_file is set."
        )

    num_cal = len(cal_seeds)

    # Score-based calibration: collect OOD score distributions via rollouts.
    if isinstance(policy, ThresholdPolicy):
        metric = config.coord_policy.metric
        if metric in ("max_prob", "max_logit", "ensemble_variance"):
            print(
                f"Generating {num_cal} calibration scores for threshold "
                f"policy with {metric} metric..."
            )
            policy.generate_scores(envs["cal"], num_cal)

    if isinstance(policy, OODPolicy) and not isinstance(policy, ThresholdPolicy):
        print(
            f"Generating {num_cal} calibration scores for "
            f"{type(policy).__name__}..."
        )
        policy.generate_scores(envs["cal"], num_cal)

    # Episode-length calibration: run weak agent alone on cal levels.
    if isinstance(
        policy,
        (TimestepRandomPolicy, ExponentialHeuristicPolicy, WaitPolicy),
    ):
        print(f"Calibrating {type(policy).__name__}: measuring episode lengths...")
        if isinstance(policy, TimestepRandomPolicy):
            old_value: Any = policy.prob
            policy.prob = 0.0  # weak agent only
        elif isinstance(policy, ExponentialHeuristicPolicy):
            old_value = 1 - policy.non_ood_starting_prob
            policy.non_ood_starting_prob = 1.0  # weak agent only
        else:
            old_value = policy.threshold
            policy.threshold = 10000  # weak agent only

        cal_envs = make_envs()
        cal_summary = evaluator.eval(
            policy,
            cal_envs,
            ["cal"],
            num_episodes=num_cal,
            close_envs=True,
        )
        mean_ep_length = cal_summary["cal"]["episode_length_mean"]

        if isinstance(policy, TimestepRandomPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.prob = old_value
        elif isinstance(policy, ExponentialHeuristicPolicy):
            policy._mean_episode_length = mean_ep_length
            policy.non_ood_starting_prob = 1 - old_value
        else:
            policy._episode_lengths = np.array(cal_summary["cal"]["episode_lengths"])
            policy.threshold = old_value

        print(f"Mean episode length (weak only): {mean_ep_length:.1f}")

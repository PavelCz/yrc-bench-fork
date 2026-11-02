"""Record a small set of coordination rollouts at one fixed threshold.

Uses the same flags, environment, policies, and GIF renderer as eval_afhp.py,
without calibration or an adaptive threshold sweep. Set
``evaluation.video_threshold`` in the YAML config. These selected examples
are illustrative rollouts, not AFHP performance estimates.
As in scripts/run_eval.py, the first query hands control to the expert for
the rest of the episode.
"""

import json
from pathlib import Path

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.level_seeds import load_level_seed_splits
from YRC.coverage.coverage_search import update_policy_params


def main() -> None:
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)
    threshold = config.evaluation.video_threshold
    if threshold is None:
        raise ValueError("Set evaluation.video_threshold in the YAML config.")
    if config.environment.common.num_envs != 1:
        raise ValueError("Use -num_envs 1 to record selected levels in order.")
    seeds = load_level_seed_splits(config, required_splits=("ood_eval",))["ood_eval"]
    num_episodes = config.algorithm.num_rollouts or len(seeds)
    if num_episodes <= 0:
        raise ValueError("-num_rollouts must be positive.")
    seeds = seeds[:num_episodes]
    if len(seeds) != num_episodes:
        raise ValueError("The seed file contains fewer levels than -num_rollouts.")
    config.evaluation.video_episodes_to_collect = num_episodes
    config.evaluation.video_filter = ["all"]
    config.evaluation.video_logging_mode = "folder"
    config.evaluation.include_human_view = True
    # Match the paper launcher: expert control persists after the first query.
    config.evaluation.defer_to_oracle = True

    envs = env_factory.make(
        config,
        # Procgen snapshots Heist's terminal counters while resetting into the
        # next level. Keep one unused seed available for that final reset;
        # stop after num_episodes, before rolling out the guard level.
        level_seeds_by_split={"test": seeds + [seeds[-1]]},
        require_level_seeds_for_splits=("test",),
    )
    try:
        policy = policy_factory.make(config, envs["test"])
        update_policy_params(policy, threshold)
        evaluator = Evaluator(config, env_config=config.environment)
        summary = evaluator.eval(
            policy,
            envs,
            ["test"],
            num_episodes=num_episodes,
            threshold=threshold,
            close_envs=False,
        )
        config_utils.save_config_json(config, Path(config.eval_run_dir) / "config.json")
        videos = []
        video_index = 0
        for worker_episodes in evaluator.episode_metadata:
            for metadata in worker_episodes:
                if metadata:
                    episode_index = metadata["episode_idx"]
                    if episode_index < len(summary["test"]["level_seeds"]):
                        videos.append(
                            {
                                "video_index": video_index,
                                "episode_index": episode_index,
                                "level_seed": summary["test"]["level_seeds"][
                                    episode_index
                                ],
                                "is_ood": bool(
                                    metadata["level_info"]["randomize_goal"]
                                ),
                                "asked_for_help": bool(
                                    metadata["level_info"]["level_ood_pred"]
                                ),
                                "return": float(
                                    metadata["episode_data"]["cumulative_reward"]
                                ),
                                "episode_length": int(
                                    metadata["episode_data"]["episode_length"]
                                ),
                            }
                        )
                video_index += 1
        output = Path(config.eval_run_dir) / "rollout_summary.json"
        output.write_text(
            json.dumps(
                config_utils.make_json_serializable(
                    {
                        "threshold": threshold,
                        "level_seeds": seeds,
                        "summary": summary,
                        "videos": videos,
                    }
                ),
                indent=2,
            )
        )
        print(f"Rollout summary: {output}")
    finally:
        for env in envs.values():
            env.close()


if __name__ == "__main__":
    main()

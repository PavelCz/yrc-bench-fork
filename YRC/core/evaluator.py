import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from pytorch_lightning.loggers import WandbLogger
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from YRC.core.video_utils import process_and_log_video, resolve_video_output_folder

class Evaluator:
    LOGGED_ACTION = 1

    # Video logging configuration constants
    VIDEO_CONFIG = {
        "fps": 10,
        "final_frame_repetitions": 10,
        "score_bar_height": 28,
        "score_bar_bg_color": 64,  # Dark gray
        "font_size": 22,
        "text_padding": 6,
        "char_width_estimate": 11,
        "normal_color": [0, 255, 0],  # Green
        "ood_color": [255, 0, 0],  # Red
        "text_color": [255, 255, 255],  # White
        "outline_color": [0, 0, 0],  # Black
    }

    def __init__(self, config, env_config: Optional[dict] = None, random_env_switch: bool = False):
        self.args = config.evaluation

        self.eval_run_dir = Path(config.eval_run_dir)

        self.collected_states = []
        self.done_saving_actions_for_vid = False
        self.video_episodes_collected = 0
        self.video_filter_passed = {}

        self.defer_to_oracle: Optional[bool] = None

        self.env_config = env_config

        self.episode_metadata: List[List[Dict]] = []

        # Check if we should skip score normalization (for max_prob metric)
        # max_prob outputs probabilities in [0, 1] range, so normalization would be misleading
        metric = getattr(config.coord_policy, "metric", None)
        alg_cls = getattr(config.algorithm, "cls", None)
        self.skip_score_normalization = (
            metric == "max_prob" or alg_cls == "RandomAlgorithm"
        )

        self.random_env_switch = random_env_switch

    def _random_env_switch_is_ood(self, env, i: int) -> bool:
        """
        Determine OOD ground-truth for random-env-switch evaluation.

        Convention:
        - env1 (venv1) is in-distribution  => OOD GT False
        - env2 (venv2) is out-of-domain    => OOD GT True

        This relies on `RandomEnvSwitchWrapper.env_selector` being present.
        We intentionally fail fast if it's missing to avoid silently producing
        incorrect OOD metrics.
        """
        base_env = getattr(env, "base_env", env)
        if not hasattr(base_env, "env_selector"):
            raise RuntimeError(
                "random_env_switch=True but the environment has no `env_selector`. "
                "Expected the underlying env to be a `RandomEnvSwitchWrapper` (or compatible). "
                "Make sure you constructed the test env via `RandomEnvSwitchWrapper(env1, env2, ...)` "
                "and passed that wrapped env into `CoordEnv` / the evaluator. "
                f"Got base_env type: {type(base_env)}"
            )

        selector = getattr(base_env, "env_selector")
        try:
            # RandomEnvSwitchWrapper uses: True => env1, False => env2
            is_env1 = bool(selector[i])
        except Exception as e:
            raise RuntimeError(
                "random_env_switch=True but could not index `env_selector` for env idx "
                f"{i}. Got env_selector type: {type(selector)}"
            ) from e

        # env2 is OOD
        return not is_env1

    def eval(
        self,
        policy,
        envs,
        eval_splits,
        num_episodes=None,
        logger: Optional[WandbLogger] = None,
        threshold: Optional[float] = None,
    ):
        args = self.args
        policy.eval()

        self.defer_to_oracle = args.defer_to_oracle

        self.done_saving_actions_for_vid = False
        self.video_episodes_collected = 0
        self.video_filter_passed = {}  # Track counts per filter
        self.collected_states: List[List[List[Dict]]] = []
        self.episode_metadata: List[
            List[Dict]
        ] = []  # Track which episodes passed each filter

        summary = {}
        for split in eval_splits:
            if num_episodes is None:
                if "val" in split:
                    num_episodes = self.env_config["val"].num_levels
                    # num_episodes = args.validation_episodes
                else:
                    assert "test" in split
                    num_episodes = self.env_config["test"].num_levels
                    # num_episodes = args.test_episodes
                assert num_episodes % envs[split].num_envs == 0

            # The dimensions of the collected states are:
            # [env_idx][episode_idx][state_idx]
            # i.e. for every parallel env, we collect a list of episodes, and for each
            # episode, we collect a list of states.
            self.collected_states: List[List[List[Dict]]] = []
            self.episode_metadata: List[List[Dict]] = []

            # Get video filters - default to ['all'] if not specified
            video_filters = getattr(args, "video_filter", ["all"])
            filter_mode = getattr(args, "video_filter_mode", "any")

            # Initialize per-filter tracking
            if filter_mode == "any":
                # Track each filter separately
                for filter_name in video_filters:
                    self.video_filter_passed[filter_name] = 0
            else:  # filter_mode == "all"
                # Track combined filter (all must pass)
                combined_filter_name = "_and_".join(video_filters)
                self.video_filter_passed[combined_filter_name] = 0

            for i in range(envs[split].num_envs):
                self.collected_states.append([])
                self.collected_states[i].append([])
                self.episode_metadata.append([])
                self.episode_metadata[i].append({})

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            log = self._eval_loop(
                policy, envs[split], num_episodes
            )

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            # Calculate AFHP for logging
            if self.defer_to_oracle:
                afhp = summary[split]["ood_pred_percentage"]
            else:
                afhp = summary[split]["action_1_frac"]

            # Create and save OOD score histograms with AFHP
            self._save_score_histograms(
                split,
                log.get("scores_in_domain", []),
                log.get("scores_out_of_domain", []),
                afhp=afhp,
                logger=logger,
                scores_original_in_domain=log.get("scores_original_in_domain", []),
                scores_original_out_of_domain=log.get(
                    "scores_original_out_of_domain", []
                ),
            )

            envs[split].close()

            # Process and log videos if logger is available
            if logger is not None:
                self._process_and_log_videos(split, threshold, afhp, logger)

                logger.experiment.log({
                    "num_finished_episodes": summary[split]["num_finished_episodes"],
                })
                if self.random_env_switch:
                    logger.experiment.log({
                    "env_1_percentage": summary[split]["num_finished_episodes_env1"] / summary[split]["num_finished_episodes"],
                })

        return summary

    def _eval_loop(self, policy, env, max_episodes: int) -> dict:
        args = self.args

        log = {
            "returns": [],
            "env_returns": [],
            "episode_length": [],
            f"action_{self.LOGGED_ACTION}": [],
            # Whether *any* state has been predicted as ood.
            "level_ood_pred": [],
            "level_ood_gt": [],
            # OOD scores per timestep for histogram analysis
            "scores_in_domain": [],  # Scores for deterministic coin levels
            "scores_out_of_domain": [],  # Scores for random coin levels
            # Original scores (before rolling average)
            "scores_original_in_domain": [],
            "scores_original_out_of_domain": [],
            # Episode outcome information
            "invisible_coin_collected": [],  # Whether coin was collected this episode
            # First timestep when OOD was predicted (None if never predicted)
            "first_ood_timestep": [],
            # Track total number of finished episodes
            "num_finished_episodes": 0,
            "num_finished_episodes_env1": 0,
            "num_finished_episodes_env2": 0,
        }

        # A temporary log that only contains stats for the current episode.
        episode_log = {
            "cumulative_reward": [0] * env.num_envs,
            "cumulative_env_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.LOGGED_ACTION}": [0] * env.num_envs,
            "randomize_goal": [0] * env.num_envs,
        }

        # For every env, whether the current level has been predicted as ood.
        current_level_ood_pred = [False] * env.num_envs
        # For every env, whether the current level is actually ood.
        current_level_ood_gt = [False] * env.num_envs
        # For every env, whether the coin was collected in the current episode.
        current_invisible_coin_collected = [False] * env.num_envs
        # For every env, the first timestep when OOD was predicted (None if never predicted)
        first_ood_timestep = [None] * env.num_envs

        obs = env.reset()
        # Initialize OOD GT for the *current* episode.
        #
        # - Coinrun-style tasks populate `info["randomize_goal"]` (handled after stepping).
        # - Random env switch tasks define OOD GT by which underlying env is selected:
        #   env1 = in-distribution, env2 = OOD.
        if self.random_env_switch:
            # RandomEnvSwitchWrapper uses a boolean env_selector:
            #   True  => venv1 (env1)
            #   False => venv2 (env2)
            # Therefore, OOD GT (env2) is the negation of env_selector.
            current_level_ood_gt = [
                self._random_env_switch_is_ood(env, i) for i in range(env.num_envs)
            ]
        prev_obs = obs
        # Track previous info for human-resolution frames (info["rgb"])
        # On first frame after reset, this will be empty dicts (no human frame available)
        prev_info = [{} for _ in range(env.num_envs)]

        for i in range(env.num_envs):
            # Maker sure there are no stale scores in the rolling average buffer.
            policy.reset_rolling_average_buffer(i)
            # Reset episode counter for heuristic policies
            if hasattr(policy, "reset_episode"):
                policy.reset_episode()

        # This tracks the very first done and is only used to determine whether to keep
        # collecting observations that are later used to generate the video.
        has_done = np.array([False] * env.num_envs)
        num_episodes = 0

        # Upper bound to prevent infinite loops if videos are hard to collect
        episode_upper_bound = max_episodes * 5

        while num_episodes < episode_upper_bound and (
            num_episodes < max_episodes or not self.done_saving_actions_for_vid
        ):
            # Add the episode timestep to the obs. This is necessary for
            # LevelBasedRandomPolicy to know whether a new episode has started.
            obs["episode_timestep"] = episode_log["episode_length"]
            # For most policies I have seen, the greedy flag is ignored. These include
            # random, lightning_ae, and ood.
            action, scores, recons = policy.act(
                obs, greedy=args.act_greedy, return_scores_and_recons=True
            )

            # Store original action before it might be modified by defer_to_oracle
            original_action = action.copy()

            # if not all(has_done):
            #     self.collected_states.append({
            #         "obs": obs["env_obs"],
            #         "scores": scores,
            #         "recons": recons,
            #         "action": action,
            #     })

            # Check whether any action is OOD.
            for i in range(env.num_envs):
                if action[i] == self.LOGGED_ACTION:
                    current_level_ood_pred[i] = True

                # Normally, we perform the action as returned by the coordination
                # policy. However, if we defer to the oracle, we perform the orcale
                # action if *any* state has been found as OOD. If no state has been
                # found as OOD, we don't perform the oracle action.
                if self.defer_to_oracle:
                    action[i] = int(current_level_ood_pred[i])

            obs, reward, done, info = env.step(action)

            for i in range(env.num_envs):
                if "env_reward" in info[i]:
                    episode_log["cumulative_env_reward"][i] += info[i]["env_reward"]

                # Track if the invisible coin was collected in this step
                if "invisible_coin_collected" in info[i]:
                    if info[i]["invisible_coin_collected"] == 1:
                        current_invisible_coin_collected[i] = True

                episode_log["cumulative_reward"][i] += reward[i]
                episode_log["episode_length"][i] += 1

                # Track the first timestep when OOD was predicted (after incrementing episode_length)
                # Use original_action to check the actual OOD prediction, not the potentially modified action
                if (
                    original_action[i] == self.LOGGED_ACTION
                    and first_ood_timestep[i] is None
                ):
                    first_ood_timestep[i] = episode_log["episode_length"][i]
                episode_log[f"action_{self.LOGGED_ACTION}"][i] += (
                    action[i] == self.LOGGED_ACTION
                ).sum()
                # Since has done is changed below, we also need to check done here.
                # Since done will not stay True, because the env is reset at the end,
                # we can't just only check done.
                # if not has_done[i] and not done[i] and not self.collected_actions_done:
                # Recons is None for non reconstruction-based OOD detectors like
                # Deep-SVDD.
                recons_i = recons[i] if recons is not None else None

                # Some OOD detectors, like the random one, don't assign scores.
                scores_i = scores[i] if scores is not None else None

                # Collect scores for histogram analysis (only if scores exist)
                if scores_i is not None and not done[i]:
                    # Check if this level has random or deterministic coin
                    is_random_coin = current_level_ood_gt[i]

                    # Collect rolling average scores (or final scores if no rolling average)
                    if is_random_coin:
                        log["scores_out_of_domain"].append(float(scores_i))
                    else:
                        log["scores_in_domain"].append(float(scores_i))

                    # Collect original scores (before rolling average) if available
                    if (
                        hasattr(policy, "last_scores_original")
                        and policy.last_scores_original is not None
                    ):
                        # Convert tensor to numpy if needed
                        original_scores = policy.last_scores_original
                        if hasattr(original_scores, "cpu"):
                            original_scores = original_scores.cpu().numpy()
                        scores_original_i = float(original_scores[i])

                        if is_random_coin:
                            log["scores_original_out_of_domain"].append(
                                scores_original_i
                            )
                        else:
                            log["scores_original_in_domain"].append(scores_original_i)

                if not self.done_saving_actions_for_vid:
                    # Get human-resolution frame from previous info if available
                    # (info["rgb"] contains 512x512 frames when render_mode="rgb_array")
                    human_obs = prev_info[i].get("rgb", None)
                    self.collected_states[i][-1].append(
                        {
                            "obs": prev_obs["env_obs"][i],
                            "human_obs": human_obs,
                            "scores": scores_i,
                            "recons": recons_i,
                            "action": action[i],
                            "done": done[i],
                        }
                    )

                if done[i]:
                    # Only log to evaluation metrics if within max_episodes limit
                    if num_episodes < max_episodes:
                        log["level_ood_pred"].append(current_level_ood_pred[i])
                        log["returns"].append(episode_log["cumulative_reward"][i])
                        log["env_returns"].append(
                            episode_log["cumulative_env_reward"][i]
                        )
                        log["episode_length"].append(episode_log["episode_length"][i])
                        log[f"action_{self.LOGGED_ACTION}"].append(
                            episode_log[f"action_{self.LOGGED_ACTION}"][i]
                        )
                        log["level_ood_gt"].append(current_level_ood_gt[i])

                        # Log episode outcome information
                        log["invisible_coin_collected"].append(
                            current_invisible_coin_collected[i]
                        )
                        # Log first OOD timestep (None if never predicted)
                        log["first_ood_timestep"].append(first_ood_timestep[i])

                    # Always increment episode counter (for upper bound check)
                    num_episodes += 1
                    # Track total finished episodes
                    log["num_finished_episodes"] += 1

                    if self.random_env_switch:
                        # Count the env for the episode that just finished.
                        # `RandomEnvSwitchWrapper` may already have sampled the NEXT
                        # episode's env_selector by the time we see done=True, so
                        # we rely on the per-episode GT we already stored.
                        #
                        # Convention:
                        # - env1 (in-distribution) => OOD GT False
                        # - env2 (OOD)             => OOD GT True
                        if current_level_ood_gt[i]:
                            log["num_finished_episodes_env2"] += 1
                        else:
                            log["num_finished_episodes_env1"] += 1

                    # Check which filters this episode passes
                    episode_data = {
                        "cumulative_reward": episode_log["cumulative_reward"][i],
                        "episode_length": episode_log["episode_length"][i],
                        "final_done": done[
                            i
                        ],  # This is the done state that caused the episode to end
                    }
                    level_info = {
                        "randomize_goal": current_level_ood_gt[
                            i
                        ],  # This will be updated below, but we capture the current value
                        "level_ood_gt": current_level_ood_gt[i],
                        "level_ood_pred": current_level_ood_pred[i],
                    }

                    # Reset the level_ood_pred for the next episode.
                    current_level_ood_pred[i] = False
                    # Reset the coin collected status for the next episode.
                    current_invisible_coin_collected[i] = False
                    # Reset the first OOD timestep for the next episode.
                    first_ood_timestep[i] = None

                    filter_results = self._check_episode_filters(
                        episode_data, level_info
                    )

                    # Count this completed episode for video collection limit
                    if not self.done_saving_actions_for_vid:
                        # Get filter mode
                        filter_mode = getattr(self.args, "video_filter_mode", "any")

                        # Track which filters this episode passed
                        passed_filters = [
                            f for f, passed in filter_results.items() if passed
                        ]

                        # Determine if episode should be kept based on filters
                        should_keep_episode = self._should_keep_episode(
                            filter_results, filter_mode
                        )

                        if should_keep_episode:
                            # Check if we need to keep this episode or if we already have enough
                            need_this_episode = self._need_episode_for_video(
                                passed_filters, filter_mode
                            )

                            if need_this_episode:
                                # Store episode metadata for later video saving
                                self.episode_metadata[i][-1] = {
                                    "filter_results": filter_results,
                                    "episode_data": episode_data,
                                    "level_info": level_info,
                                    "episode_idx": num_episodes
                                    - 1,  # Global episode index
                                }

                                # Update per-filter counts
                                self._update_video_filter_counts(
                                    passed_filters, filter_mode, num_episodes
                                )

                                self.video_episodes_collected += 1

                                # Create a new list for the next episode for this env.
                                self.collected_states[i].append([])
                                self.episode_metadata[i].append({})
                            else:
                                # Episode passes filters but we have enough - discard
                                self.collected_states[i][-1] = []
                                self.episode_metadata[i][-1] = {}
                        else:
                            # Episode didn't pass filters - discard collected states
                            # Clear the current episode data and reuse the same slot
                            self.collected_states[i][-1] = []
                            self.episode_metadata[i][-1] = {}

                    episode_log["cumulative_reward"][i] = 0
                    episode_log["cumulative_env_reward"][i] = 0
                    episode_log["episode_length"][i] = 0
                    episode_log[f"action_{self.LOGGED_ACTION}"][i] = 0

                    # In case we are using a rolling average for the score, we need to
                    # reset the buffer for the next episode.
                    policy.reset_rolling_average_buffer(i)
                    # Reset episode counter for heuristic policies
                    if hasattr(policy, "reset_episode"):
                        policy.reset_episode()

                # We update this after we (potentially) save this to log. This is
                # because gym3 automatically resets the environment at the end of an
                # episode, so the info dict might be of the next episode.
                if self.random_env_switch:
                    # In random env switch mode, GT is determined by which underlying
                    # env is selected for the (possibly just-reset) current episode.
                    current_level_ood_gt[i] = self._random_env_switch_is_ood(env, i)
                else:
                    # Coinrun-style GT: randomize_goal indicates OOD.
                    current_level_ood_gt[i] = bool(info[i]["randomize_goal"])

                # Check if all filters have enough episodes
                if self._all_video_filters_satisfied():
                    self.done_saving_actions_for_vid = True
            prev_obs = obs
            # Update prev_info for human-resolution frames on next iteration
            prev_info = [info[i] for i in range(env.num_envs)]

            has_done |= done

        return log

    def summarize(self, log):
        total_steps = int(sum(log["episode_length"]))
        ood_pred_percentage = float(np.mean(log["level_ood_pred"]))
        ood_accuracy = float(np.mean(log["level_ood_pred"] == log["level_ood_gt"]))
        return {
            "steps": total_steps,
            "num_finished_episodes": log["num_finished_episodes"],
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "episode_lengths": log["episode_length"],  # Raw episode lengths per episode
            "return_mean": float(np.mean(log["returns"])),
            "raw_returns": log["returns"],
            "return_std": float(np.std(log["returns"])),
            "env_return_mean": float(np.mean(log["env_returns"])),
            "env_return_std": float(np.std(log["env_returns"])),
            f"action_{self.LOGGED_ACTION}_frac": float(
                sum(log[f"action_{self.LOGGED_ACTION}"]) / total_steps
            ),
            "ood_pred_percentage": ood_pred_percentage,
            "ood_accuracy": ood_accuracy,
            "level_ood_gt": log["level_ood_gt"],
            "level_ood_pred": log["level_ood_pred"],
            # Episode outcome information
            "invisible_coin_collected": log["invisible_coin_collected"],
            "first_ood_timestep": log["first_ood_timestep"],
            "num_finished_episodes_env1": log["num_finished_episodes_env1"],
            "num_finished_episodes_env2": log["num_finished_episodes_env2"],
        }

    def write_summary(self, split, summary):
        log_str = f"   Steps:       {summary['steps']}\n"
        log_str += f"   Finished Episodes: {summary['num_finished_episodes']}\n"
        log_str += "   Episode:    "
        log_str += f"mean {summary['episode_length_mean']:7.2f}  "
        log_str += f"min {summary['episode_length_min']:7.2f}  "
        log_str += f"max {summary['episode_length_max']:7.2f}\n"
        log_str += "   Reward:     "
        log_str += f"mean {summary['return_mean']:.2f} "
        log_str += f"± {(1.96 * summary['return_std']) / (len(summary['raw_returns']) ** 0.5):.2f}\n"
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_return_mean']:.2f} "
        log_str += f"± {(1.96 * summary['env_return_std']) / (len(summary['raw_returns']) ** 0.5):.2f}\n"
        log_str += f"   Action {self.LOGGED_ACTION} fraction: {summary[f'action_{self.LOGGED_ACTION}_frac']:7.2f}\n"
        log_str += f"   OOD Pred Percentage: {summary['ood_pred_percentage']:7.2f}\n"
        log_str += f"   OOD Accuracy: {summary['ood_accuracy']:7.2f}\n"
        log_str += "   Raw Rewards: "
        for r in summary["raw_returns"]:
            log_str += f"{r:.2f},"
        logging.info(log_str)

        return summary

    def _save_score_histograms(
        self,
        split: str,
        scores_in_domain: List[float],
        scores_out_of_domain: List[float],
        afhp: float,
        logger: Optional[WandbLogger] = None,
        scores_original_in_domain: Optional[List[float]] = None,
        scores_original_out_of_domain: Optional[List[float]] = None,
    ) -> None:
        """Create and save histograms of OOD scores for in-domain and out-of-domain levels.

        If original scores are provided, creates 4 histograms (2 for original, 2 for rolling avg).
        Otherwise, creates 2 histograms (1 for in-domain, 1 for out-of-domain).
        """
        if not scores_in_domain and not scores_out_of_domain:
            logging.info("No scores available for histogram generation")
            return

        # Filter out infinite values
        def filter_finite(scores):
            """Remove positive and negative infinity from scores."""
            finite_scores = [s for s in scores if np.isfinite(s)]
            return finite_scores

        # Determine if we have rolling average (original scores provided)
        has_rolling_avg = (
            scores_original_in_domain is not None and len(scores_original_in_domain) > 0
        ) or (
            scores_original_out_of_domain is not None
            and len(scores_original_out_of_domain) > 0
        )

        if has_rolling_avg:
            # Create figure with 4 subplots (2 rows, 2 columns)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

            # Filter scores
            scores_original_in_domain_filtered = filter_finite(
                scores_original_in_domain or []
            )
            scores_original_out_of_domain_filtered = filter_finite(
                scores_original_out_of_domain or []
            )
            scores_in_domain_filtered = filter_finite(scores_in_domain)
            scores_out_of_domain_filtered = filter_finite(scores_out_of_domain)

            # Log filtered values
            if scores_original_in_domain:
                n_filtered = len(scores_original_in_domain) - len(
                    scores_original_in_domain_filtered
                )
                if n_filtered > 0:
                    logging.info(
                        f"Filtered {n_filtered} infinite values from original in-domain scores"
                    )
            if scores_original_out_of_domain:
                n_filtered = len(scores_original_out_of_domain) - len(
                    scores_original_out_of_domain_filtered
                )
                if n_filtered > 0:
                    logging.info(
                        f"Filtered {n_filtered} infinite values from original out-of-domain scores"
                    )
            if len(scores_in_domain) != len(scores_in_domain_filtered):
                logging.info(
                    f"Filtered {len(scores_in_domain) - len(scores_in_domain_filtered)} infinite values from rolling avg in-domain scores"
                )
            if len(scores_out_of_domain) != len(scores_out_of_domain_filtered):
                logging.info(
                    f"Filtered {len(scores_out_of_domain) - len(scores_out_of_domain_filtered)} infinite values from rolling avg out-of-domain scores"
                )

            # Top row: Original scores (before rolling average)
            # Histogram for original in-domain levels (deterministic coin)
            if scores_original_in_domain_filtered:
                ax1.hist(
                    scores_original_in_domain_filtered,
                    bins=50,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
                ax1.set_xlabel("OOD Score (Original)")
                ax1.set_ylabel("Frequency")
                ax1.set_title(
                    f"Original Scores - In-Domain (Deterministic Coin)\n{split} - {len(scores_original_in_domain_filtered)} samples"
                )
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(
                    0.5, 0.5, "No original in-domain scores", ha="center", va="center"
                )
                ax1.set_title(f"Original Scores - In-Domain\n{split} - No data")

            # Histogram for original out-of-domain levels (random coin)
            if scores_original_out_of_domain_filtered:
                ax2.hist(
                    scores_original_out_of_domain_filtered,
                    bins=50,
                    alpha=0.7,
                    color="red",
                    edgecolor="black",
                )
                ax2.set_xlabel("OOD Score (Original)")
                ax2.set_ylabel("Frequency")
                ax2.set_title(
                    f"Original Scores - Out-of-Domain (Random Coin)\n{split} - {len(scores_original_out_of_domain_filtered)} samples"
                )
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No original out-of-domain scores",
                    ha="center",
                    va="center",
                )
                ax2.set_title(f"Original Scores - Out-of-Domain\n{split} - No data")

            # Bottom row: Rolling average scores
            # Histogram for rolling avg in-domain levels (deterministic coin)
            if scores_in_domain_filtered:
                ax3.hist(
                    scores_in_domain_filtered,
                    bins=50,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
                ax3.set_xlabel("OOD Score (Rolling Avg)")
                ax3.set_ylabel("Frequency")
                ax3.set_title(
                    f"Rolling Avg Scores - In-Domain (Deterministic Coin)\n{split} - {len(scores_in_domain_filtered)} samples - AFHP: {afhp:.2f}"
                )
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No rolling avg in-domain scores",
                    ha="center",
                    va="center",
                )
                ax3.set_title(
                    f"Rolling Avg Scores - In-Domain\n{split} - No data - AFHP: {afhp:.2f}"
                )

            # Histogram for rolling avg out-of-domain levels (random coin)
            if scores_out_of_domain_filtered:
                ax4.hist(
                    scores_out_of_domain_filtered,
                    bins=50,
                    alpha=0.7,
                    color="red",
                    edgecolor="black",
                )
                ax4.set_xlabel("OOD Score (Rolling Avg)")
                ax4.set_ylabel("Frequency")
                ax4.set_title(
                    f"Rolling Avg Scores - Out-of-Domain (Random Coin)\n{split} - {len(scores_out_of_domain_filtered)} samples - AFHP: {afhp:.2f}"
                )
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No rolling avg out-of-domain scores",
                    ha="center",
                    va="center",
                )
                ax4.set_title(
                    f"Rolling Avg Scores - Out-of-Domain\n{split} - No data - AFHP: {afhp:.2f}"
                )
        else:
            # Create figure with 2 subplots (no rolling average)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            scores_in_domain_filtered = filter_finite(scores_in_domain)
            scores_out_of_domain_filtered = filter_finite(scores_out_of_domain)

            # Log if we filtered out any values
            if len(scores_in_domain) != len(scores_in_domain_filtered):
                logging.info(
                    f"Filtered {len(scores_in_domain) - len(scores_in_domain_filtered)} infinite values from in-domain scores"
                )
            if len(scores_out_of_domain) != len(scores_out_of_domain_filtered):
                logging.info(
                    f"Filtered {len(scores_out_of_domain) - len(scores_out_of_domain_filtered)} infinite values from out-of-domain scores"
                )

            # Histogram for in-domain levels (deterministic coin)
            if scores_in_domain_filtered:
                ax1.hist(
                    scores_in_domain_filtered,
                    bins=50,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
                ax1.set_xlabel("OOD Score")
                ax1.set_ylabel("Frequency")
                ax1.set_title(
                    f"OOD Scores - In-Domain Levels (Deterministic Coin)\n{split} - {len(scores_in_domain_filtered)} samples (finite) - AFHP: {afhp:.2f}"
                )
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, "No in-domain scores", ha="center", va="center")
                ax1.set_title(
                    f"OOD Scores - In-Domain Levels\n{split} - No data - AFHP: {afhp:.2f}"
                )

            # Histogram for out-of-domain levels (random coin)
            if scores_out_of_domain_filtered:
                ax2.hist(
                    scores_out_of_domain_filtered,
                    bins=50,
                    alpha=0.7,
                    color="red",
                    edgecolor="black",
                )
                ax2.set_xlabel("OOD Score")
                ax2.set_ylabel("Frequency")
                ax2.set_title(
                    f"OOD Scores - Out-of-Domain Levels (Random Coin)\n{split} - {len(scores_out_of_domain_filtered)} samples (finite) - AFHP: {afhp:.2f}"
                )
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No out-of-domain scores", ha="center", va="center")
                ax2.set_title(
                    f"OOD Scores - Out-of-Domain Levels\n{split} - No data - AFHP: {afhp:.2f}"
                )

        plt.tight_layout()

        # Save to file with AFHP in filename
        histogram_path = (
            self.eval_run_dir / f"ood_score_histograms_{split}_afhp_{afhp:.2f}.png"
        )
        plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved OOD score histograms to {histogram_path}")

        # Log to wandb if available with AFHP in the key
        if logger is not None:
            import wandb

            logger.experiment.log(
                {
                    f"ood_score_histograms_{split}_afhp_{afhp:.2f}": wandb.Image(
                        str(histogram_path)
                    )
                }
            )

        plt.close(fig)

    # Helper methods for video filtering and processing
    def _check_episode_filters(
        self, episode_data: dict, level_info: dict
    ) -> Dict[str, bool]:
        """Check which filters an episode passes. Returns dict mapping filter names to boolean results."""
        video_filters = getattr(self.args, "video_filter", ["all"])
        results = {}

        # Extract episode information
        total_reward = episode_data.get("cumulative_reward", 0)
        # episode_length = episode_data.get("episode_length", 0)
        randomize_goal = level_info.get("randomize_goal", False)
        # level_ood_gt = level_info.get("level_ood_gt", False)
        level_ood_pred = level_info.get("level_ood_pred", False)
        # final_done_state = episode_data.get("final_done", False)

        for filter_type in video_filters:
            if filter_type == "all":
                results[filter_type] = True
            elif filter_type == "random_coin_success":
                # Agent successfully got the coin (positive reward) AND coin was randomly placed
                results[filter_type] = total_reward > 0 and randomize_goal
            elif filter_type == "deterministic_coin_success":
                # Agent successfully got the coin (positive reward) AND coin was deterministically placed
                results[filter_type] = total_reward > 0 and not randomize_goal
            elif filter_type == "ood_detected":
                # OOD was detected in this episode
                results[filter_type] = level_ood_pred
            elif filter_type == "in_distribution":
                # In-distribution episode (no OOD detected)
                results[filter_type] = not level_ood_pred
            else:
                # Unknown filter type, default to saving
                results[filter_type] = True

        return results

    def _should_keep_episode(
        self, filter_results: Dict[str, bool], filter_mode: str
    ) -> bool:
        """Determine if an episode should be kept based on filter results and mode.

        Args:
            filter_results: Dict mapping filter names to whether the episode passed that filter
            filter_mode: Either "any" (keep if passes at least one filter) or "all" (keep only if passes all filters)

        Returns:
            True if episode should be kept, False otherwise
        """
        if filter_mode == "any":
            # Keep if passes at least one filter
            return any(filter_results.values())
        else:  # filter_mode == "all"
            # Keep only if passes all filters
            return all(filter_results.values())

    def _need_episode_for_video(
        self, passed_filters: List[str], filter_mode: str
    ) -> bool:
        """Check if we still need to collect videos for any of the passed filters.

        Args:
            passed_filters: List of filter names that this episode passed
            filter_mode: Either "any" or "all"

        Returns:
            True if we need this episode for video collection, False if we have enough
        """
        if filter_mode == "any":
            # Check if any filter still needs more episodes
            for filter_name in passed_filters:
                if filter_name in self.video_filter_passed:
                    if (
                        self.video_filter_passed[filter_name]
                        < self.args.video_episodes_to_collect
                    ):
                        return True
            return False
        else:  # filter_mode == "all"
            # Check if combined filter still needs more episodes
            video_filters = getattr(self.args, "video_filter", ["all"])
            combined_filter_name = "_and_".join(video_filters)
            if combined_filter_name in self.video_filter_passed:
                return (
                    self.video_filter_passed[combined_filter_name]
                    < self.args.video_episodes_to_collect
                )
            return False

    def _update_video_filter_counts(
        self, passed_filters: List[str], filter_mode: str, num_episodes: int
    ) -> None:
        """Update the count of collected videos for each filter.

        Args:
            passed_filters: List of filter names that this episode passed
            filter_mode: Either "any" or "all"
            num_episodes: Current episode number (for logging)
        """
        if filter_mode == "any":
            # Track each filter separately
            for filter_name in passed_filters:
                if filter_name in self.video_filter_passed:
                    self.video_filter_passed[filter_name] += 1
        else:  # filter_mode == "all"
            # Track combined filter (only if all passed)
            video_filters = getattr(self.args, "video_filter", ["all"])
            combined_filter_name = "_and_".join(video_filters)
            if combined_filter_name in self.video_filter_passed:
                self.video_filter_passed[combined_filter_name] += 1

        # Log progress for each filter
        for filter_name, count in self.video_filter_passed.items():
            logging.info(
                f"Episode {num_episodes} - Filter '{filter_name}': {count}/{self.args.video_episodes_to_collect}"
            )

    def _all_video_filters_satisfied(self) -> bool:
        """Check if all video filters have collected enough episodes.

        Returns:
            True if all filters have enough episodes, False otherwise
        """
        if self.args.video_episodes_to_collect <= 0:
            return False

        for filter_name in self.video_filter_passed.keys():
            if (
                self.video_filter_passed[filter_name]
                < self.args.video_episodes_to_collect
            ):
                return False

        return True

    def _process_and_log_videos(
        self,
        split: str,
        threshold: Optional[float],
        afhp: float,
        logger: WandbLogger,
    ) -> None:
        """Process collected episode states and log them as videos.

        Args:
            split: The evaluation split (e.g., "val", "test")
            threshold: The OOD threshold value
            afhp: The AFHP (Agent-Friendly Help Probability) value
            logger: WandB logger for logging videos
        """
        args = self.args

        # Determine output folder for video logging
        raw_output_folder = getattr(args, "video_output_folder", None)
        logging_mode = getattr(args, "video_logging_mode", "none")

        if raw_output_folder is None and logging_mode in ["folder", "both"]:
            output_folder = self._get_default_video_folder()
        elif raw_output_folder is not None and logging_mode in ["folder", "both"]:
            output_folder = resolve_video_output_folder(
                raw_output_folder, self.eval_run_dir, create_folder=True
            )
        else:
            # For wandb/none modes, don't create folders even if specified
            output_folder = None

        # Get video filters for this evaluation
        video_filters = getattr(args, "video_filter", ["all"])
        filter_mode = getattr(args, "video_filter_mode", "any")
        max_videos = args.video_episodes_to_collect

        # Track how many videos we've logged per filter
        videos_logged = {}
        if filter_mode == "any":
            for filter_name in video_filters:
                videos_logged[filter_name] = 0
        else:
            combined_filter_name = "_and_".join(video_filters)
            videos_logged[combined_filter_name] = 0

        # Global episode index for video logging.
        global_episode_idx = 0

        for env_idx in range(len(self.collected_states)):
            for episode_idx in range(len(self.collected_states[env_idx])):
                episode: List[Dict] = self.collected_states[env_idx][episode_idx]

                # Only log videos for completed episodes.
                if len(episode) > 0 and episode[-1]["done"]:
                    # Get stored episode metadata
                    episode_meta = self.episode_metadata[env_idx][episode_idx]

                    if episode_meta:  # Check if metadata exists
                        filter_results = episode_meta["filter_results"]

                        if filter_mode == "any":
                            # Save video to appropriate filter folders (separate categories)
                            for filter_name, passed in filter_results.items():
                                # Only save if this filter passed AND we haven't reached the limit
                                if passed and videos_logged[filter_name] < max_videos:
                                    # Create filter-specific output folder
                                    filter_output_folder = None
                                    if output_folder is not None:
                                        filter_output_folder = (
                                            output_folder / filter_name
                                        )
                                        filter_output_folder.mkdir(
                                            parents=True, exist_ok=True
                                        )

                                    # Save video for this filter
                                    process_and_log_video(
                                        episode,
                                        global_episode_idx,
                                        threshold,
                                        afhp,
                                        self.VIDEO_CONFIG,
                                        output_folder=filter_output_folder,
                                        logger=logger,
                                        logging_mode=logging_mode,
                                        subfolder=filter_name,
                                        wandb_category=f"videos_{filter_name}",
                                        skip_score_normalization=self.skip_score_normalization,
                                    )
                                    videos_logged[filter_name] += 1
                        else:  # filter_mode == "all"
                            # Only save if ALL filters passed AND we haven't reached the limit
                            if (
                                all(filter_results.values())
                                and videos_logged[combined_filter_name] < max_videos
                            ):
                                filter_output_folder = None
                                if output_folder is not None:
                                    filter_output_folder = (
                                        output_folder / combined_filter_name
                                    )
                                    filter_output_folder.mkdir(
                                        parents=True, exist_ok=True
                                    )

                                # Save video to combined filter folder
                                process_and_log_video(
                                    episode,
                                    global_episode_idx,
                                    threshold,
                                    afhp,
                                    self.VIDEO_CONFIG,
                                    output_folder=filter_output_folder,
                                    logger=logger,
                                    logging_mode=logging_mode,
                                    subfolder=combined_filter_name,
                                    wandb_category=f"videos_{combined_filter_name}",
                                    skip_score_normalization=self.skip_score_normalization,
                                )
                                videos_logged[combined_filter_name] += 1

                global_episode_idx += 1

    def _get_default_video_folder(self) -> Path:
        """Get or create the default video folder in the eval_run_dir or experiment
        directory."""
        # Try to get eval_run_dir from config first
        base_dir = self.eval_run_dir

        video_folder = base_dir / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        return video_folder

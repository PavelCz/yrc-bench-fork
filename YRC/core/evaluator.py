import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from pytorch_lightning.loggers import WandbLogger

from YRC.core.video_utils import process_and_log_video, resolve_video_output_folder


class Evaluator:
    LOGGED_ACTION = 1

    # Video logging configuration constants
    VIDEO_CONFIG = {
        "fps": 10,
        "final_frame_repetitions": 10,
        "score_bar_height": 15,
        "score_bar_bg_color": 64,  # Dark gray
        "font_size": 12,
        "text_padding": 5,
        "char_width_estimate": 6,
        "normal_color": [0, 255, 0],  # Green
        "ood_color": [255, 0, 0],  # Red
        "text_color": [255, 255, 255],  # White
        "outline_color": [0, 0, 0],  # Black
    }

    def __init__(self, config, env_config: Optional[dict] = None):
        self.args = config.evaluation

        self.eval_run_dir = Path(config.eval_run_dir)

        self.collected_states = []
        self.done_saving_actions_for_vid = False
        self.video_episodes_collected = 0
        self.video_filter_passed = {}

        self.defer_to_oracle: Optional[bool] = None

        self.env_config = env_config

        self.episode_metadata: List[List[Dict]] = []

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
        final_done_state = episode_data.get("final_done", False)

        for filter_type in video_filters:
            if filter_type == "all":
                results[filter_type] = True
            elif filter_type == "no_death":
                # Episode did not end by the agent dying (assuming death means done=True at the end)
                # For procgen coinrun, death typically means the agent didn't collect the coin in time
                results[
                    filter_type
                ] = not final_done_state  # Don't save if episode ended with done=True
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

            # Initialize per-filter tracking
            for filter_name in video_filters:
                self.video_filter_passed[filter_name] = 0

            for i in range(envs[split].num_envs):
                self.collected_states.append([])
                self.collected_states[i].append([])
                self.episode_metadata.append([])
                self.episode_metadata[i].append({})

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            log = self._eval_loop(policy, envs[split], num_episodes)

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            envs[split].close()

            if logger is not None:
                # Determine output folder for video logging
                raw_output_folder = getattr(self.args, "video_output_folder", None)
                logging_mode = getattr(self.args, "video_logging_mode", "none")

                if raw_output_folder is None and logging_mode in ["folder", "both"]:
                    output_folder = self._get_default_video_folder()
                elif raw_output_folder is not None and logging_mode in [
                    "folder",
                    "both",
                ]:
                    output_folder = resolve_video_output_folder(
                        raw_output_folder, self.eval_run_dir, create_folder=True
                    )
                else:
                    # For wandb/none modes, don't create folders even if specified
                    output_folder = None

                # Choose correct AFHP for video logging
                if self.defer_to_oracle:
                    afhp = summary[split]["ood_pred_percentage"]
                else:
                    afhp = summary[split]["action_1_frac"]

                # Get video filters for this evaluation
                video_filters = getattr(args, "video_filter", ["all"])

                # Global episode index for video logging.
                global_episode_idx = 0

                for env_idx in range(len(self.collected_states)):
                    for episode_idx in range(len(self.collected_states[env_idx])):
                        episode: List[Dict] = self.collected_states[env_idx][
                            episode_idx
                        ]

                        # Only log videos for completed episodes.
                        if len(episode) > 0 and episode[-1]["done"]:
                            # Get stored episode metadata
                            episode_meta = self.episode_metadata[env_idx][episode_idx]

                            if episode_meta:  # Check if metadata exists
                                filter_results = episode_meta["filter_results"]

                                # Save video to appropriate filter folders
                                for filter_name, passed in filter_results.items():
                                    if passed:
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
                                        )

                        global_episode_idx += 1
        return summary

    def _get_default_video_folder(self) -> Path:
        """Get or create the default video folder in the eval_run_dir or experiment
        directory."""
        # Try to get eval_run_dir from config first
        base_dir = self.eval_run_dir

        video_folder = base_dir / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        return video_folder

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

        obs = env.reset()
        prev_obs = obs

        # This tracks the very first done and is only used to determine whether to keep
        # collecting observations that are later used to generate the video.
        has_done = np.array([False] * env.num_envs)
        num_episodes = 0

        episode_upper_bound = max_episodes * 5

        while (num_episodes < max_episodes or not self.done_saving_actions_for_vid) and num_episodes < episode_upper_bound:
            # Add the episode timestep to the obs. This is necessary for
            # OneCheckRandomPolicy to know whether a new episode has started.
            obs["episode_timestep"] = episode_log["episode_length"]
            # For most policies I have seen, the greedy flag is ignored. These include
            # random, lightning_ae, and ood.
            action, scores, recons = policy.act(
                obs, greedy=args.act_greedy, return_scores_and_recons=True
            )

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

                episode_log["cumulative_reward"][i] += reward[i]
                episode_log["episode_length"][i] += 1
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

                if not self.done_saving_actions_for_vid:
                    self.collected_states[i][-1].append(
                        {
                            "obs": prev_obs["env_obs"][i],
                            "scores": scores_i,
                            "recons": recons_i,
                            "action": action[i],
                            "done": done[i],
                        }
                    )

                if done[i] and num_episodes < max_episodes:
                    log["level_ood_pred"].append(current_level_ood_pred[i])
                    log["returns"].append(episode_log["cumulative_reward"][i])
                    log["env_returns"].append(episode_log["cumulative_env_reward"][i])
                    log["episode_length"].append(episode_log["episode_length"][i])
                    log[f"action_{self.LOGGED_ACTION}"].append(
                        episode_log[f"action_{self.LOGGED_ACTION}"][i]
                    )
                    log["level_ood_gt"].append(current_level_ood_gt[i])
                    num_episodes += 1

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

                    filter_results = self._check_episode_filters(
                        episode_data, level_info
                    )

                    # Count this completed episode for video collection limit
                    if not self.done_saving_actions_for_vid:
                        # Track which filters this episode passed
                        passed_filters = [
                            f for f, passed in filter_results.items() if passed
                        ]

                        # Store episode metadata for later video saving
                        self.episode_metadata[i][-1] = {
                            "filter_results": filter_results,
                            "episode_data": episode_data,
                            "level_info": level_info,
                            "episode_idx": num_episodes - 1,  # Global episode index
                        }

                        # Update per-filter counts
                        for filter_name in passed_filters:
                            self.video_filter_passed[filter_name] += 1

                        # Log progress for each filter
                        for filter_name, count in self.video_filter_passed.items():
                            logging.info(
                                f"Episode {num_episodes} - Filter '{filter_name}': {count}/{self.args.video_episodes_to_collect}"
                            )

                        self.video_episodes_collected += 1

                        # Create a new list for the next episode for this env.
                        self.collected_states[i].append([])
                        self.episode_metadata[i].append({})

                    episode_log["cumulative_reward"][i] = 0
                    episode_log["cumulative_env_reward"][i] = 0
                    episode_log["episode_length"][i] = 0
                    episode_log[f"action_{self.LOGGED_ACTION}"][i] = 0

                    # In case we are using a rolling average for the score, we need to
                    # reset the buffer for the next episode.
                    policy.reset_rolling_average_buffer(i)

                # We update this after we (potentially) save this to log. This is
                # because gym3 automatically resets the environment at the end of an
                # episode, so the info dict might be of the next episode.
                current_level_ood_gt[i] = info[i]["randomize_goal"]

                # Check if all filters have enough episodes
                video_filters = getattr(self.args, "video_filter", ["all"])
                all_filters_satisfied = True
                for filter_name in video_filters:
                    if (
                        self.video_filter_passed[filter_name]
                        < self.args.video_episodes_to_collect
                    ):
                        all_filters_satisfied = False
                        break

                if all_filters_satisfied and self.args.video_episodes_to_collect > 0:
                    self.done_saving_actions_for_vid = True
            prev_obs = obs

            has_done |= done

        return log

    def summarize(self, log):
        total_steps = int(sum(log["episode_length"]))
        ood_pred_percentage = float(np.mean(log["level_ood_pred"]))
        ood_accuracy = float(np.mean(log["level_ood_pred"] == log["level_ood_gt"]))
        return {
            "steps": total_steps,
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
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
        }

    def write_summary(self, split, summary):
        log_str = f"   Steps:       {summary['steps']}\n"
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

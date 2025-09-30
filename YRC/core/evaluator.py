import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from pytorch_lightning.loggers import WandbLogger
import wandb

from YRC.core.video_utils import process_and_log_video, resolve_video_output_folder
from YRC.core.configs.global_configs import get_global_variable


class Evaluator:
    LOGGED_ACTION = 1

    # Video logging configuration constants
    VIDEO_CONFIG = {
        'fps': 10,
        'final_frame_repetitions': 10,
        'score_bar_height': 15,
        'score_bar_bg_color': 64,  # Dark gray
        'font_size': 12,
        'text_padding': 5,
        'char_width_estimate': 6,
        'normal_color': [0, 255, 0],   # Green
        'ood_color': [255, 0, 0],      # Red
        'text_color': [255, 255, 255], # White
        'outline_color': [0, 0, 0],    # Black
    }

    def __init__(self, config, env_config: Optional[dict] = None):
        self.args = config.evaluation

        self.eval_run_dir = Path(config.eval_run_dir)

        self.collected_states = []
        self.collected_actions_done = False

        self.defer_to_oracle: Optional[bool] = None

        self.env_config = env_config

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

        self.collected_actions_done = False
        self.collected_states: List[List[Dict]] = []

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

            self.collected_states = []
            for i in range(envs[split].num_envs):
                self.collected_states.append([])

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            log = self._eval_loop(policy, envs[split], num_episodes)

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            envs[split].close()

            if logger is not None:
                afhp = summary[split]["action_1_frac"]
                # Determine output folder for video logging
                raw_output_folder = getattr(self.args, 'video_output_folder', None)
                logging_mode = getattr(self.args, 'video_logging_mode', 'none')

                if raw_output_folder is None and logging_mode in ['folder', 'both']:
                    output_folder = self._get_default_video_folder()
                elif raw_output_folder is not None and logging_mode in ['folder', 'both']:
                    output_folder = resolve_video_output_folder(raw_output_folder, self.eval_run_dir, create_folder=True)
                else:
                    # For wandb/none modes, don't create folders even if specified
                    output_folder = None

                for i in range(len(self.collected_states)):
                    process_and_log_video(
                        self.collected_states, i, logger, threshold, afhp, self.VIDEO_CONFIG,
                        output_folder=output_folder,
                        logging_mode=logging_mode
                    )

        return summary

    def _get_default_video_folder(self) -> Path:
        """Get or create the default video folder in the eval_run_dir or experiment directory."""
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

        while num_episodes < max_episodes:
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

                # Since has done is changed below, we also need to check done here.
                # Since done will not stay True, because the env is reset at the end,
                # we can't just only check done.
                if not has_done[i] and not done[i] and not self.collected_actions_done:
                    # Recons is None for non reconstruction-based OOD detectors like
                    # Deep-SVDD.
                    recons_i = recons[i] if recons is not None else None

                    # Some OOD detectors, like the random one, don't assign scores.
                    scores_i = scores[i] if scores is not None else None

                    self.collected_states[i].append(
                        {
                            "obs": prev_obs["env_obs"][i],
                            "scores": scores_i,
                            "recons": recons_i,
                            "action": action[i],
                        }
                    )
            prev_obs = obs

            has_done |= done

            if all(has_done):
                self.collected_actions_done = True

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

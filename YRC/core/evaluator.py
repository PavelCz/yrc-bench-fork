import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image, ImageDraw, ImageFont


class VideoProcessor:
    """Handles video frame processing and manipulation operations."""

    def __init__(self, config: dict):
        self.config = config

    def combine_observations_and_reconstructions(
        self, observations: np.ndarray, reconstructions: np.ndarray
    ) -> np.ndarray:
        """Combine observations and reconstructions side by side."""
        obs_vid = np.stack(observations, axis=0)

        if reconstructions[0] is not None:
            recons_vid = np.stack(reconstructions, axis=0)

            # Ensure both videos have the same shape
            if obs_vid.shape != recons_vid.shape:
                recons_vid = np.resize(recons_vid, obs_vid.shape)

            # Clip the reconstructions to the range 0-1
            recons_vid = np.clip(recons_vid, 0, 1)

            # Concatenate horizontally (side by side)
            combined_vid = np.concatenate([obs_vid, recons_vid], axis=-1)
        else:
            combined_vid = obs_vid

        # Normalize to 0-255 range
        combined_vid = combined_vid * 255
        combined_vid = combined_vid.astype(np.uint8)

        return combined_vid

    def add_repeated_frames(self, video: np.ndarray) -> np.ndarray:
        """Add repeated frames at the end for smoother video ending."""
        final_frame = video[-1:].copy()
        repeated_frames = np.repeat(final_frame, self.config['final_frame_repetitions'], axis=0)
        return np.concatenate([video, repeated_frames], axis=0)

    def create_video_with_bars(self, video: np.ndarray) -> np.ndarray:
        """Create a new video array with space for score bars."""
        time_steps, channels, height, width = video.shape
        bar_height = self.config['score_bar_height']

        return np.zeros(
            (time_steps, channels, height + bar_height, width), dtype=np.uint8
        )

    def add_base_video_content(self, video_with_bars: np.ndarray, original_video: np.ndarray) -> None:
        """Copy original video content below the bar area."""
        bar_height = self.config['score_bar_height']
        video_with_bars[:, :, bar_height:, :] = original_video


class ScoreBarRenderer:
    """Handles score bar rendering and color logic."""

    def __init__(self, config: dict):
        self.config = config

    def calculate_score_bounds(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate min and max score bounds, handling -inf values."""
        all_scores = np.array(scores)
        all_rational_scores = all_scores[all_scores != float("-inf")]
        score_min = np.min(all_rational_scores)
        score_max = np.max(all_scores)
        return score_min, score_max

    def normalize_score(self, score: float, score_min: float, score_max: float) -> float:
        """Normalize a score to 0-1 range."""
        if score_max > score_min:
            normalized_score = (score - score_min) / (score_max - score_min)
            # Clamp edge cases
            return np.clip(normalized_score, 0, 1)
        else:
            return 0.5

    def get_bar_color(self, action: int) -> List[int]:
        """Get the appropriate color for the score bar based on action."""
        if action == 1:  # OOD detected
            return self.config['ood_color']  # Red
        else:  # Normal
            return self.config['normal_color']  # Green

    def calculate_bar_dimensions(self, normalized_score: float, width: int) -> Tuple[int, int]:
        """Calculate bar width and determine if background should be filled."""
        bar_width = int(normalized_score * width)
        return bar_width, bar_width < width

    @staticmethod
    def calculate_score_bounds_static(scores: List[float]) -> Tuple[float, float]:
        """Static method to calculate min and max score bounds, handling -inf values."""
        all_scores = np.array(scores)
        all_rational_scores = all_scores[all_scores != float("-inf")]
        score_min = np.min(all_rational_scores)
        score_max = np.max(all_scores)
        return score_min, score_max


class TextRenderer:
    """Handles text rendering and PIL image operations."""

    def __init__(self, config: dict):
        self.config = config
        self.font = self._load_font()

    def _load_font(self) -> Optional[ImageFont.FreeTypeFont]:
        """Load font with fallback options."""
        try:
            return ImageFont.truetype("arial.ttf", self.config['font_size'])
        except OSError:
            try:
                return ImageFont.load_default()
            except OSError:
                return None

    def calculate_text_dimensions(self, text: str) -> Tuple[int, int]:
        """Calculate text width and height."""
        if self.font:
            bbox = self.font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            # Fallback estimation
            return len(text) * self.config['char_width_estimate'], self.config['font_size'] + 2

    def calculate_text_position(self, bar_height: int, text_height: int) -> Tuple[int, int]:
        """Calculate text position within the bar."""
        text_x = self.config['text_padding']
        text_y = (bar_height - text_height) // 2
        return text_x, text_y

    def add_text_to_frame(self, frame: np.ndarray, text: str, position: Tuple[int, int]) -> np.ndarray:
        """Add text overlay to a video frame."""
        # Convert from (C, H, W) to (H, W, C)
        rgb_frame = frame.transpose(1, 2, 0)
        pil_image = Image.fromarray(rgb_frame, mode="RGB")
        draw = ImageDraw.Draw(pil_image)

        text_x, text_y = position

        if self.font:
            # Draw black outline
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text(
                            (text_x + dx, text_y + dy),
                            text,
                            fill=tuple(self.config['outline_color']),
                            font=self.font,
                        )
            # Draw white text
            draw.text(
                (text_x, text_y),
                text,
                fill=tuple(self.config['text_color']),
                font=self.font
            )
        else:
            # Fallback without font
            draw.text((text_x, text_y), text, fill=tuple(self.config['text_color']))

        # Convert back to (C, H, W)
        return np.array(pil_image).transpose(2, 0, 1)


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

    def __init__(self, config, env_config: dict):
        self.args = config
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
                for i in range(len(self.collected_states)):
                    self._log_evaluation_video(logger, threshold, afhp, i)

        return summary

    def extract_video_data(self, episode_idx: int) -> Optional[Dict[str, Any]]:
        """Extract and validate video data from collected states."""
        if episode_idx >= len(self.collected_states) or not self.collected_states[episode_idx]:
            logging.warning(f"No data available for episode {episode_idx}")
            return None

        episode_data = self.collected_states[episode_idx]
        obs = [x["obs"] for x in episode_data]
        scores = [x["scores"] for x in episode_data]
        recons = [x["recons"] for x in episode_data]
        actions = [x["action"] for x in episode_data]

        return {
            'observations': obs,
            'scores': scores[0] if scores and scores[0] is not None else None,
            'reconstructions': recons,
            'actions': actions
        }

    def add_score_bars(
        self,
        video: np.ndarray,
        scores: List[float],
        actions: List[int],
        score_renderer: ScoreBarRenderer,
        text_renderer: TextRenderer
    ) -> np.ndarray:
        """Add score bars to video frames."""
        score_min, score_max = score_renderer.calculate_score_bounds(scores)
        bar_height = self.VIDEO_CONFIG['score_bar_height']
        time_steps, channels, _, width = video.shape

        # Create new video with extra height for score bar
        vid_with_bars = self.create_video_with_bars(video)
        self.add_base_video_content(vid_with_bars, video)

        # Add score bars for each frame
        for t in range(time_steps):
            # Get current score and action, handling repeated frames
            if t < len(scores):
                current_score = scores[t]
                current_action = actions[t] if t < len(actions) else 0
            else:
                # For repeated frames, use the last values
                current_score = scores[-1] if scores else 0.0
                current_action = actions[-1] if actions else 0

            # Normalize score to 0-1 range
            normalized_score = score_renderer.normalize_score(current_score, score_min, score_max)

            # Calculate bar width
            bar_width, needs_bg = score_renderer.calculate_bar_dimensions(normalized_score, width)
            bar_color = score_renderer.get_bar_color(current_action)

            # Fill the bar area
            if bar_width > 0:
                vid_with_bars[t, :, :bar_height, :bar_width] = np.array(bar_color)[:, np.newaxis, np.newaxis]

            # Add background for remaining part of bar (dark gray)
            if needs_bg:
                vid_with_bars[t, :, :bar_height, bar_width:] = self.VIDEO_CONFIG['score_bar_bg_color']

            # Add text overlay with score value
            score_text = f"{current_score:.3f}"
            text_width, text_height = text_renderer.calculate_text_dimensions(score_text)
            text_x, text_y = text_renderer.calculate_text_position(bar_height, text_height)

            vid_with_bars[t] = text_renderer.add_text_to_frame(
                vid_with_bars[t], score_text, (text_x, text_y)
            )

        return vid_with_bars

    def generate_caption(self, threshold: float, afhp: float, video_data: Dict[str, Any]) -> str:
        """Generate video caption with relevant information."""
        caption = f"Threshold: {threshold:.2E} - AFHP: {afhp:.2f}"

        use_recons = video_data['reconstructions'][0] is not None
        if use_recons:
            caption += " - Left: Original, Right: Reconstruction"
        else:
            caption += " - Original observations"

        if video_data['scores'] is not None:
            score_min, score_max = ScoreBarRenderer.calculate_score_bounds_static(video_data['scores'])
            caption += (
                " - Top bar: Score with values (Green=Normal, Red=OOD, Range: "
                f"{score_min:.3f}-{score_max:.3f})"
            )

        return caption

    def log_to_wandb(self, logger: WandbLogger, video: np.ndarray, caption: str, afhp: float) -> None:
        """Log video to WandB."""
        logger.experiment.log({
            f"eval_episode_{afhp:.2f}": wandb.Video(
                video,
                fps=self.VIDEO_CONFIG['fps'],
                format="gif",
                caption=caption,
            ),
        })

    def create_video_with_bars(self, video: np.ndarray) -> np.ndarray:
        """Create a new video array with space for score bars."""
        processor = VideoProcessor(self.VIDEO_CONFIG)
        return processor.create_video_with_bars(video)

    def add_base_video_content(self, video_with_bars: np.ndarray, original_video: np.ndarray) -> None:
        """Copy original video content below the bar area."""
        processor = VideoProcessor(self.VIDEO_CONFIG)
        processor.add_base_video_content(video_with_bars, original_video)

    def _log_evaluation_video(
        self, logger: WandbLogger, threshold: float, afhp: float, episode_idx: int
    ) -> None:
        """
        Generate and log evaluation video with score bars to wandb.

        Args:
            logger: WandbLogger instance for logging
            threshold: Threshold value for the video caption
            afhp: Ask for help percentage
            episode_idx: Index of the episode to log
        """
        # Extract and validate data
        video_data = self.extract_video_data(episode_idx)
        if not video_data:
            return

        # Create video processor instance
        processor = VideoProcessor(self.VIDEO_CONFIG)

        # Process video frames
        combined_video = processor.combine_observations_and_reconstructions(
            video_data['observations'],
            video_data['reconstructions']
        )

        combined_video = processor.add_repeated_frames(combined_video)

        # Add score bars if available
        if video_data['scores'] is not None:
            score_renderer = ScoreBarRenderer(self.VIDEO_CONFIG)
            text_renderer = TextRenderer(self.VIDEO_CONFIG)

            combined_video = self.add_score_bars(
                combined_video,
                video_data['scores'],
                video_data['actions'],
                score_renderer,
                text_renderer
            )

        # Generate caption and log
        caption = self.generate_caption(threshold, afhp, video_data)
        self.log_to_wandb(logger, combined_video, caption, afhp)

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

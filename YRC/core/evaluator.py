import logging
import numpy as np
from typing import Optional
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image, ImageDraw, ImageFont

class Evaluator:
    LOGGED_ACTION = 1

    def __init__(self, config):
        self.args = config
        self.collected_states = []
        self.collected_actions_done = False

    def eval(
        self, 
        policy, 
        envs, 
        eval_splits, 
        num_episodes=None, 
        logger: Optional[WandbLogger] = None, 
        threshold: Optional[float] = None,
        percentile_step: Optional[float] = None,
    ):
        args = self.args
        policy.eval()

        self.collected_actions_done = False
        self.collected_states = []

        summary = {}
        for split in eval_splits:
            if num_episodes is None:
                if "val" in split:
                    num_episodes = args.validation_episodes
                else:
                    assert "test" in split
                    num_episodes = args.test_episodes
                assert num_episodes % envs[split].num_envs == 0

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            log = self._eval_loop(policy, envs[split], num_episodes)

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            envs[split].close()

        if logger is not None:
            self._log_evaluation_video(logger, threshold)

        return summary

    def _log_evaluation_video(self, logger: WandbLogger, threshold: float) -> None:
        """
        Generate and log evaluation video with score bars to wandb.
        
        Args:
            logger: WandbLogger instance for logging
            threshold: Threshold value for the video caption
        """
        obs = [x["obs"] for x in self.collected_states]
        scores = [x["scores"] for x in self.collected_states]
        recons = [x["recons"] for x in self.collected_states]
        action = [x["action"] for x in self.collected_states]

        # We determine whether our OOD detector uses reconstructions by checking
        # whether the first element of the first reconstruction is None.
        use_recons = recons[0][0] is not None

        # Stack observations and reconstructions
        obs_vid = np.stack(obs, axis=1)

        if use_recons:
            
            recons_vid = np.stack(recons, axis=1)
            
            # Ensure both videos have the same shape
            if obs_vid.shape != recons_vid.shape:
                # Reshape reconstructions to match observations if needed
                recons_vid = np.resize(recons_vid, obs_vid.shape)
            
            # Concatenate horizontally (side by side)
            # obs_vid and recons_vid have shape (batch, time, c, h, w)
            # We want to concatenate along the width dimension (last dimension)
            combined_vid = np.concatenate([obs_vid, recons_vid], axis=-1)
        else:
            combined_vid = obs_vid
        
        # Normalize to 0-255 range
        combined_vid = combined_vid * 255
        combined_vid = combined_vid.astype(np.uint8)
        
        # Add score bars at the top of each frame
        # Find global min and max scores across all frames
        all_scores = np.concatenate(scores)
        score_min = np.min(all_scores)
        score_max = np.max(all_scores)
        
        # Create score bar visualization
        bar_height = 20  # Height of the score bar in pixels
        batch_size, time_steps, channels, height, width = combined_vid.shape
        
        # Create new video with extra height for score bar
        vid_with_bars = np.zeros((batch_size, time_steps, channels, height + bar_height, width), dtype=np.uint8)
        
        # Copy original video content below the bar area
        vid_with_bars[:, :, :, bar_height:, :] = combined_vid
        
        # Add score bars for each frame
        for t in range(time_steps):
            for b in range(batch_size):
                if t < len(scores) and b < len(scores[t]):
                    current_score = scores[t][b]
                    
                    # Normalize score to 0-1 range
                    if score_max > score_min:
                        normalized_score = (current_score - score_min) / (score_max - score_min)
                    else:
                        normalized_score = 0.5
                    
                    # Calculate bar width (as fraction of total width)
                    bar_width = int(normalized_score * width)
                    
                    # Create score bar (red for high scores, green for low scores)
                    bar_color = [
                        int(255 * normalized_score),      # Red channel
                        int(255 * (1 - normalized_score)), # Green channel  
                        0                                  # Blue channel
                    ]
                    
                    # Fill the bar area
                    if bar_width > 0:
                        vid_with_bars[b, t, :, :bar_height, :bar_width] = np.array(bar_color)[:, np.newaxis, np.newaxis]
                    
                    # Add background for remaining part of bar (dark gray)
                    if bar_width < width:
                        vid_with_bars[b, t, :, :bar_height, bar_width:] = 64  # Dark gray
                    
                    # Add text overlay with score value
                    # Convert the current frame to PIL Image for text rendering
                    frame = vid_with_bars[b, t].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                    pil_image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pil_image)
                    
                    # Try to use a small font, fall back to default if not available
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except:
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                    
                    # Format score text
                    score_text = f"{current_score:.3f}"
                    
                    # Calculate text position (fixed location)
                    if font:
                        text_bbox = draw.textbbox((0, 0), score_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    else:
                        text_width = len(score_text) * 6  # Rough estimate
                        text_height = 11
                    
                    # Fixed position: left side of the bar with small padding
                    text_x = 5
                    text_y = (bar_height - text_height) // 2
                    
                    # Draw text with white color and black outline for better visibility
                    if font:
                        # Black outline
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx != 0 or dy != 0:
                                    draw.text((text_x + dx, text_y + dy), score_text, fill=(0, 0, 0), font=font)
                        # White text
                        draw.text((text_x, text_y), score_text, fill=(255, 255, 255), font=font)
                    else:
                        # Fallback without font
                        draw.text((text_x, text_y), score_text, fill=(255, 255, 255))
                    
                    # Convert back to numpy array
                    frame_with_text = np.array(pil_image).transpose(2, 0, 1)  # Convert back to (C, H, W)
                    vid_with_bars[b, t] = frame_with_text
        
        combined_vid = vid_with_bars
        
        logger.experiment.log(
            {
                f"eval_episode_{threshold:.2f}": wandb.Video(
                    # (batch dim, time dim, c, h, w)
                    combined_vid,
                    fps=15,
                    format="gif",
                    caption=(
                        f"Threshold: {threshold:.2f} - "
                        f"{'Left: Original, Right: Reconstruction' if use_recons else 'Original observations'} - "
                        f"Top bar: Score with values (Green=Low, Red=High, Range: {score_min:.3f}-{score_max:.3f})"
                    ),
                ),
            }
        )

    def _eval_loop(self, policy, env, max_episodes: int) -> dict:
        args = self.args

        log = {
            "reward": [],
            "env_reward": [],
            "episode_length": [],
            f"action_{self.LOGGED_ACTION}": [],
        }

        # A temporary log that only contains stats for the current episode.
        episode_log = {
            "reward": [0] * env.num_envs,
            "env_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.LOGGED_ACTION}": [0] * env.num_envs,
        }

        obs = env.reset()

        # This tracks the very first done and is only used to determine whether to keep
        # collecting observations that are later used to generate the video.
        has_done = np.array([False] * env.num_envs)
        num_episodes = 0

        while num_episodes < max_episodes:

            # For most policies I have seen, the greedy flag is ignored. These include
            # random, lightning_ae, and ood.
            action, scores, recons = policy.act(
                obs, greedy=args.act_greedy, return_scores_and_recons=True
            )

            if not all(has_done):
                self.collected_states.append({
                    "obs": obs["env_obs"],
                    "scores": scores,
                    "recons": recons,
                    "action": action,
                })

            obs, reward, done, info = env.step(action)

            for i in range(env.num_envs):

                if "env_reward" in info[i]:
                    episode_log["env_reward"][i] += info[i]["env_reward"]

                episode_log["reward"][i] += reward[i]
                episode_log["episode_length"][i] += 1
                episode_log[f"action_{self.LOGGED_ACTION}"][i] += (action[i] == self.LOGGED_ACTION).sum()

                if done[i] and num_episodes < max_episodes:
                    log["reward"].append(episode_log["reward"][i])
                    log["env_reward"].append(episode_log["env_reward"][i])
                    log["episode_length"].append(episode_log["episode_length"][i])
                    log[f"action_{self.LOGGED_ACTION}"].append(episode_log[f"action_{self.LOGGED_ACTION}"][i])
                    num_episodes += 1

                    episode_log["reward"][i] = 0
                    episode_log["env_reward"][i] = 0
                    episode_log["episode_length"][i] = 0
                    episode_log[f"action_{self.LOGGED_ACTION}"][i] = 0

            has_done |= done

        return log

    def summarize(self, log):
        total_steps = int(sum(log["episode_length"]))
        return {
            "steps": total_steps,
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "reward_mean": float(np.mean(log["reward"])),
            "raw_reward": log["reward"],
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            f"action_{self.LOGGED_ACTION}_frac": float(
                sum(log[f"action_{self.LOGGED_ACTION}"]) / total_steps
            ),
        }

    def write_summary(self, split, summary):
        log_str = f"   Steps:       {summary['steps']}\n"
        log_str += "   Episode:    "
        log_str += f"mean {summary['episode_length_mean']:7.2f}  "
        log_str += f"min {summary['episode_length_min']:7.2f}  "
        log_str += f"max {summary['episode_length_max']:7.2f}\n"
        log_str += "   Reward:     "
        log_str += f"mean {summary['reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['env_reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += f"   Action {self.LOGGED_ACTION} fraction: {summary[f'action_{self.LOGGED_ACTION}_frac']:7.2f}\n"
        log_str += "   Raw Rewards: "
        for r in summary["raw_reward"]:
            log_str += f"{r:.2f},"
        logging.info(log_str)

        return summary

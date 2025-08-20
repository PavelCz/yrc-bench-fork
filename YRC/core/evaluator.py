import logging
import numpy as np
from typing import Optional, List, Dict
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
    ):
        args = self.args
        policy.eval()

        self.collected_actions_done = False
        self.collected_states: List[List[Dict]] = []

        summary = {}
        for split in eval_splits:
            if num_episodes is None:
                if "val" in split:
                    num_episodes = args.validation_episodes
                else:
                    assert "test" in split
                    num_episodes = args.test_episodes
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

    def _log_evaluation_video(self, logger: WandbLogger, threshold: float, afhp: float, i: int) -> None:
        """
        Generate and log evaluation video with score bars to wandb.
        
        Args:
            logger: WandbLogger instance for logging
            threshold: Threshold value for the video caption
            afhp: Ask for help percentage
            i: Index of the episode to log
        """
        obs = [x["obs"] for x in self.collected_states[i]]
        scores = [x["scores"] for x in self.collected_states[i]]
        recons = [x["recons"] for x in self.collected_states[i]]
        actions = [x["action"] for x in self.collected_states[i]]

        # We determine whether our OOD detector uses reconstructions by checking
        # whether the first element of the first reconstruction is None.
        use_recons = recons[0] is not None

        # Stack observations and reconstructions
        obs_vid = np.stack(obs, axis=0)

        if use_recons:
            
            recons_vid = np.stack(recons, axis=0)
            
            # Ensure both videos have the same shape
            if obs_vid.shape != recons_vid.shape:
                # Reshape reconstructions to match observations if needed
                recons_vid = np.resize(recons_vid, obs_vid.shape)

            # Clip the reconstructions to the range 0-1.
            recons_vid = np.clip(recons_vid, 0, 1)
            
            # Concatenate horizontally (side by side)
            # obs_vid and recons_vid have shape (batch, time, c, h, w)
            # We want to concatenate along the width dimension (last dimension)
            combined_vid = np.concatenate([obs_vid, recons_vid], axis=-1)
        else:
            combined_vid = obs_vid
        
        # Normalize to 0-255 range
        combined_vid = combined_vid * 255
        combined_vid = combined_vid.astype(np.uint8)

        # Repeat the final frame for 1 second (10 frames at 10 fps)
        final_frame = combined_vid[-1:].copy()  # Get the last frame
        repeated_frames = np.repeat(final_frame, 10, axis=0)  # Repeat 10 times
        combined_vid = np.concatenate([combined_vid, repeated_frames], axis=0)

        use_score_bars = scores[0] is not None

        if use_score_bars:
            
            # Add score bars at the top of each frame
            # Find global min and max scores across all frames
            all_scores = np.array(scores)
            score_min = np.min(all_scores)
            score_max = np.max(all_scores)
            
            # Create score bar visualization
            bar_height = 15  # Height of the score bar in pixels
            time_steps, channels, height, width = combined_vid.shape
            
            # Create new video with extra height for score bar
            vid_with_bars = np.zeros((time_steps, channels, height + bar_height, width), dtype=np.uint8)
            
            # Copy original video content below the bar area
            vid_with_bars[:, :, bar_height:, :] = combined_vid
            
            # Add score bars for each frame
            for t in range(time_steps):
                if t < len(scores):
                    current_score = scores[t]
                    
                    # Normalize score to 0-1 range
                    if score_max > score_min:
                        normalized_score = (current_score - score_min) / (score_max - score_min)
                    else:
                        normalized_score = 0.5
                    
                    # Calculate bar width (as fraction of total width)
                    bar_width = int(normalized_score * width)
                    
                    # Create score bar (green by default, red if action is 1)
                    if t < len(actions) and actions[t] == 1:
                        # Red for action = 1 (OOD detected)
                        bar_color = [255, 0, 0]  # Red
                    else:
                        # Green for action = 0 (normal)
                        bar_color = [0, 255, 0]  # Green
                    
                    # Fill the bar area
                    if bar_width > 0:
                        vid_with_bars[t, :, :bar_height, :bar_width] = np.array(bar_color)[:, np.newaxis, np.newaxis]
                    
                    # Add background for remaining part of bar (dark gray)
                    if bar_width < width:
                        vid_with_bars[t, :, :bar_height, bar_width:] = 64  # Dark gray
                    
                    # Add text overlay with score value
                    # Convert the current frame to PIL Image for text rendering
                    frame = vid_with_bars[t].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                    pil_image = Image.fromarray(frame, mode="RGB")
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
                    vid_with_bars[t] = frame_with_text
            
            combined_vid = vid_with_bars

            caption = f"Threshold: {threshold:.2E} - AFHP: {afhp:.2f}"

            if use_recons:
                caption += " - Left: Original, Right: Reconstruction"
            else:
                caption += " - Original observations"

            if use_score_bars:
                caption += f" - Top bar: Score with values (Green=Normal, Red=OOD, Range: {score_min:.3f}-{score_max:.3f})"
                
            # # We want to log separate videos and not in a batch.
            # for i in range(combined_vid.shape[0]):

            logger.experiment.log(
                {
                    f"eval_episode_{afhp:.2f}": wandb.Video(
                        # (time dim, c, h, w)
                        combined_vid,
                        fps=10,
                        format="gif",
                        caption=caption,
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
        prev_obs = obs

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

            # if not all(has_done):
            #     self.collected_states.append({
            #         "obs": obs["env_obs"],
            #         "scores": scores,
            #         "recons": recons,
            #         "action": action,
            #     })

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

                # Since has done is changed below, we also need to check done here.
                # Since done will not stay True, because the env is reset at the end,
                # we can't just only check done.
                if not has_done[i] and not done[i] and not self.collected_actions_done:
                    self.collected_states[i].append({
                        "obs": prev_obs["env_obs"][i],
                        "scores": scores[i],
                        "recons": recons[i],
                        "action": action[i],
                    })
            prev_obs = obs

            has_done |= done

            if all(has_done):
                self.collected_actions_done = True

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

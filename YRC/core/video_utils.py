"""
Consolidated video utilities for evaluation video generation.

This module contains classes and functions for:
- Processing video frames and combining observations with reconstructions
- Rendering score bars and text overlays
- Extracting episode video data and logging videos to WandB
"""

from pathlib import Path
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any, Literal, Union
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image, ImageDraw, ImageFont

# Logger for video profiling
video_logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame processing and manipulation operations."""

    def __init__(self, config: dict):
        """
        Initialize VideoProcessor with configuration.

        Args:
            config: Dictionary containing video processing configuration
        """
        self.config = config

    def combine_observations_and_reconstructions(
        self, observations: List[np.ndarray], reconstructions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Combine observations and reconstructions side by side.

        Args:
            observations: List of observation frames
            reconstructions: List of reconstruction frames

        Returns:
            Combined video array with observations and reconstructions side by side
        """
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

    def combine_agent_and_human_views(
        self,
        agent_video: np.ndarray,
        human_observations: List[Optional[np.ndarray]],
    ) -> np.ndarray:
        """
        Combine agent view (top) with human-resolution view (bottom) vertically.

        The agent video (without bars) is upscaled to match the human view width.
        No bars are added here - bars should be added separately after combining.

        Args:
            agent_video: Agent observation video array (T, C, H, W), already in uint8 0-255.
                         Should NOT have bars - bars are added after combining.
            human_observations: List of human-resolution frames (H, W, C) or None for each timestep

        Returns:
            Combined video array with agent view on top, human view on bottom
        """
        t_start = time.perf_counter()
        time_steps, channels, agent_height, agent_width = agent_video.shape

        # Check if we have any human observations
        has_human_obs = any(h is not None for h in human_observations)
        if not has_human_obs:
            return agent_video

        # Get human frame dimensions from first non-None frame
        human_height, human_width = None, None
        for h in human_observations:
            if h is not None:
                # Human obs comes as (H, W, C), we need to get dimensions
                human_height, human_width = h.shape[0], h.shape[1]
                break

        if human_height is None:
            return agent_video

        # Scale agent to match human width for consistent stacking
        scale_factor = human_width / agent_width
        scaled_agent_height = int(agent_height * scale_factor)

        # Create output array: [scaled_agent][human]
        total_height = scaled_agent_height + human_height
        combined_vid = np.zeros(
            (time_steps, channels, total_height, human_width), dtype=np.uint8
        )

        resize_time = 0.0
        copy_time = 0.0

        for t in range(time_steps):
            t0 = time.perf_counter()

            # Resize agent frame to match human width
            agent_frame = agent_video[t].transpose(1, 2, 0)  # (H, W, C)
            agent_pil = Image.fromarray(agent_frame, mode="RGB")
            agent_resized = agent_pil.resize(
                (human_width, scaled_agent_height), Image.Resampling.NEAREST
            )
            agent_resized_np = np.array(agent_resized).transpose(2, 0, 1)  # (C, H, W)

            resize_time += time.perf_counter() - t0

            # Place frames in combined video
            t0 = time.perf_counter()

            # 1. Place agent content at top
            combined_vid[t, :, :scaled_agent_height, :] = agent_resized_np

            # 2. Place human frame below
            human_idx = min(t, len(human_observations) - 1)
            human_frame = human_observations[human_idx]

            if human_frame is not None:
                # Human frame is (H, W, C), convert to (C, H, W)
                human_frame_chw = human_frame.transpose(2, 0, 1)
                combined_vid[t, :, scaled_agent_height:, :] = human_frame_chw
            else:
                # If no human frame available, fill with gray
                combined_vid[t, :, scaled_agent_height:, :] = 128
            copy_time += time.perf_counter() - t0

        total_time = time.perf_counter() - t_start
        video_logger.debug(
            f"combine_agent_and_human_views: total={total_time:.3f}s, "
            f"resize={resize_time:.3f}s, copy={copy_time:.3f}s, frames={time_steps}"
        )

        return combined_vid

    def add_repeated_frames(self, video: np.ndarray) -> np.ndarray:
        """
        Add repeated frames at the end for smoother video ending.

        Args:
            video: Input video array

        Returns:
            Video with repeated final frames appended
        """
        final_frame = video[-1:].copy()
        repeated_frames = np.repeat(
            final_frame, self.config["final_frame_repetitions"], axis=0
        )
        return np.concatenate([video, repeated_frames], axis=0)

    def calculate_bar_height(self, video_height: int) -> int:
        """Calculate bar height based on video dimensions and configured ratio."""
        ratio = self.config.get("score_bar_height_ratio", 0.08)
        bar_height = int(video_height * ratio)
        # Ensure minimum bar height of 16 pixels for readability
        return max(16, bar_height)

    def create_video_with_bars(self, video: np.ndarray, bar_height: int) -> np.ndarray:
        """
        Create a new video array with space for score bars.

        Args:
            video: Input video array
            bar_height: Height of the bar to add

        Returns:
            New video array with extra height for score bars
        """
        time_steps, channels, height, width = video.shape

        return np.zeros(
            (time_steps, channels, height + bar_height, width), dtype=np.uint8
        )

    def add_base_video_content(
        self, video_with_bars: np.ndarray, original_video: np.ndarray, bar_height: int
    ) -> None:
        """
        Copy original video content below the bar area.

        Args:
            video_with_bars: Video array with space for bars
            original_video: Original video content to copy
            bar_height: Height of the bar area
        """
        video_with_bars[:, :, bar_height:, :] = original_video

    def upscale_video(self, video: np.ndarray, target_size: int) -> np.ndarray:
        """
        Upscale video to target size while maintaining aspect ratio.

        Args:
            video: Input video array (T, C, H, W)
            target_size: Target size for the larger dimension

        Returns:
            Upscaled video array
        """
        time_steps, channels, height, width = video.shape

        # Calculate scale factor to reach target size
        scale = target_size / max(height, width)
        if scale <= 1.0:
            return video  # Already at or above target size

        new_height = int(height * scale)
        new_width = int(width * scale)

        # Create output array
        upscaled = np.zeros(
            (time_steps, channels, new_height, new_width), dtype=np.uint8
        )

        for t in range(time_steps):
            frame = video[t].transpose(1, 2, 0)  # (H, W, C)
            pil_frame = Image.fromarray(frame, mode="RGB")
            resized = pil_frame.resize(
                (new_width, new_height), Image.Resampling.NEAREST
            )
            upscaled[t] = np.array(resized).transpose(2, 0, 1)  # (C, H, W)

        return upscaled


class ScoreBarRenderer:
    """Handles score bar rendering and color logic."""

    def __init__(self, config: dict):
        self.config = config

    def calculate_score_bounds(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate min and max score bounds, handling -inf values."""
        all_scores = np.array(scores)
        all_rational_scores = all_scores[all_scores != float("-inf")]

        # If all scores are -inf, return 0.0, 1.0
        if len(all_rational_scores) == 0:
            return 0.0, 1.0
        score_min = np.min(all_rational_scores)
        score_max = np.max(all_scores)
        return score_min, score_max

    def normalize_score(
        self, score: float, score_min: float, score_max: float
    ) -> float:
        """Normalize a score to 0-1 range."""
        if score_max > score_min:
            normalized_score = (score - score_min) / (score_max - score_min)
            return float(np.clip(normalized_score, 0, 1))
        return 0.5

    def get_bar_color(self, action: int) -> List[int]:
        """Get the appropriate color for the score bar based on action."""
        if action == 1:  # OOD detected
            return self.config["ood_color"]
        return self.config["normal_color"]

    def calculate_bar_dimensions(
        self, normalized_score: float, width: int
    ) -> Tuple[int, bool]:
        """Calculate bar width and determine if background should be filled."""
        bar_width = int(normalized_score * width)
        return bar_width, bar_width < width

    @staticmethod
    def calculate_score_bounds_static(scores: List[float]) -> Tuple[float, float]:
        """Static method to calculate min and max score bounds, handling -inf values."""
        all_scores = np.array(scores)
        all_rational_scores = all_scores[all_scores != float("-inf")]

        # If all scores are -inf, return 0.0, 1.0
        if len(all_rational_scores) == 0:
            return 0.0, 1.0

        score_min = np.min(all_rational_scores)
        score_max = np.max(all_scores)
        return score_min, score_max


class TextRenderer:
    """Handles text rendering and PIL image operations."""

    def __init__(self, config: dict, font_size: int = 16):
        """
        Initialize TextRenderer with configuration.

        Args:
            config: Dictionary containing rendering configuration
            font_size: Font size in pixels (computed from bar height)
        """
        self.config = config
        self.font_size = font_size
        self.font = self._load_font()

    def _load_font(self) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        try:
            return ImageFont.truetype("arial.ttf", self.font_size)
        except OSError:
            return ImageFont.load_default()

    def calculate_text_dimensions(self, text: str) -> Tuple[int, int]:
        if self.font:
            bbox = self.font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Fallback estimate based on font size
        char_width = max(1, self.font_size // 2)
        return len(text) * char_width, self.font_size + 2

    def calculate_text_position(
        self, bar_height: int, text_height: int
    ) -> Tuple[int, int]:
        text_x = self.config["text_padding"]
        text_y = (bar_height - text_height) // 2
        return text_x, text_y

    def add_text_to_frame(
        self, frame: np.ndarray, text: str, position: Tuple[int, int]
    ) -> np.ndarray:
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
                            fill=tuple(self.config["outline_color"]),
                            font=self.font,
                        )
            # Draw main text
            draw.text(
                (text_x, text_y),
                text,
                fill=tuple(self.config["text_color"]),
                font=self.font,
            )
        else:
            draw.text((text_x, text_y), text, fill=tuple(self.config["text_color"]))

        # Convert back to (C, H, W)
        return np.array(pil_image).transpose(2, 0, 1)


def extract_video_data(episode: List[Dict]) -> Dict[str, Any]:
    """Extract and validate video data from collected states."""

    obs = [x["obs"] for x in episode]
    scores = [x["scores"] for x in episode]
    recons = [x["recons"] for x in episode]
    actions = [x["action"] for x in episode]
    dones = [x["done"] for x in episode]
    # Extract human-resolution observations if available
    human_obs = [x.get("human_obs", None) for x in episode]

    return {
        "observations": obs,
        "human_observations": human_obs,
        "scores": scores,
        "reconstructions": recons,
        "actions": actions,
        "dones": dones,
    }


def add_score_bars(
    video: np.ndarray,
    scores: List[float],
    actions: List[int],
    score_renderer: ScoreBarRenderer,
    video_config: dict,
    skip_normalization: bool = False,
    bar_height: Optional[int] = None,
    total_episode_frames: Optional[int] = None,
    dones: Optional[List[bool]] = None,
) -> np.ndarray:
    """Add score bars to video frames.

    Args:
        video: Video array
        scores: List of scores for each frame
        actions: List of actions for each frame
        score_renderer: ScoreBarRenderer instance
        video_config: Video configuration dictionary
        skip_normalization: If True, don't normalize scores (useful for max_prob which is already in [0,1])
        bar_height: Height of the bar in pixels. If None, computed from video dimensions.
        total_episode_frames: Total number of frames in the episode (for frame counter display).
                              If None, uses len(scores).
        dones: List of done flags for each frame (for debugging episode boundaries).

    Returns:
        Video with score bars added
    """
    time_steps, channels, height, width = video.shape

    # Calculate bar height from video dimensions if not provided
    processor = VideoProcessor(video_config)
    if bar_height is None:
        bar_height = processor.calculate_bar_height(height)

    # Create text renderer with font size proportional to bar height (increased from 0.7 to 0.85)
    font_size = max(12, int(bar_height * 0.85))
    text_renderer = TextRenderer(video_config, font_size=font_size)

    # Calculate score bounds for normalization (unless we're skipping normalization)
    if not skip_normalization:
        score_min, score_max = score_renderer.calculate_score_bounds(scores)
    else:
        # For max_prob, scores are already in [0, 1] range
        score_min, score_max = 0.0, 1.0

    # Create new video with extra height for score bar
    vid_with_bars = processor.create_video_with_bars(video, bar_height)
    processor.add_base_video_content(vid_with_bars, video, bar_height)

    # Total frames for display (excluding repeated frames at end)
    total_frames = (
        total_episode_frames if total_episode_frames is not None else len(scores)
    )

    # Add score bars for each frame
    for t in range(time_steps):
        # Get current score and action, handling repeated frames
        if t < len(scores):
            current_score = scores[t]
            current_action = actions[t] if t < len(actions) else 0
            current_done = dones[t] if dones and t < len(dones) else False
            frame_num = t + 1  # 1-indexed for display
        else:
            # For repeated frames, use the last values
            current_score = scores[-1] if scores else 0.0
            current_action = actions[-1] if actions else 0
            current_done = dones[-1] if dones else False
            frame_num = total_frames  # Show last frame number for repeated frames

        # Normalize score to 0-1 range (or use as-is if skipping normalization)
        if skip_normalization:
            # Clamp to [0, 1] just in case
            normalized_score = float(np.clip(current_score, 0, 1))
        else:
            normalized_score = score_renderer.normalize_score(
                current_score, score_min, score_max
            )

        # Calculate bar width
        bar_width, needs_bg = score_renderer.calculate_bar_dimensions(
            normalized_score, width
        )
        bar_color = score_renderer.get_bar_color(current_action)

        # Fill the bar area
        if bar_width > 0:
            vid_with_bars[t, :, :bar_height, :bar_width] = np.array(bar_color)[
                :, np.newaxis, np.newaxis
            ]

        # Add background for remaining part of bar (dark gray)
        if needs_bg:
            vid_with_bars[t, :, :bar_height, bar_width:] = video_config[
                "score_bar_bg_color"
            ]

        # Add text overlay with frame counter, done status, and score value
        done_marker = " [DONE]" if current_done else ""
        bar_text = f"F{frame_num}/{total_frames}{done_marker} | {current_score:.3f}"
        _text_width, text_height = text_renderer.calculate_text_dimensions(bar_text)
        text_x, text_y = text_renderer.calculate_text_position(bar_height, text_height)

        vid_with_bars[t] = text_renderer.add_text_to_frame(
            vid_with_bars[t], bar_text, (text_x, text_y)
        )

    return vid_with_bars


def add_action_indicator_bars(
    video: np.ndarray,
    actions: List[int],
    score_renderer: ScoreBarRenderer,
    video_config: dict,
    bar_height: Optional[int] = None,
    total_episode_frames: Optional[int] = None,
    dones: Optional[List[bool]] = None,
) -> np.ndarray:
    """Add action indicator bars to video frames when scores are not available.

    Args:
        video: Video array
        actions: List of actions for each frame
        score_renderer: ScoreBarRenderer instance
        video_config: Video configuration dictionary
        bar_height: Height of the bar in pixels. If None, computed from video dimensions.
        total_episode_frames: Total number of frames in the episode (for frame counter display).
                              If None, uses len(actions).
        dones: List of done flags for each frame (for debugging episode boundaries).

    Returns:
        Video with action indicator bars added
    """
    time_steps, channels, height, width = video.shape

    # Calculate bar height from video dimensions if not provided
    processor = VideoProcessor(video_config)
    if bar_height is None:
        bar_height = processor.calculate_bar_height(height)

    # Create text renderer with font size proportional to bar height (increased from 0.7 to 0.85)
    font_size = max(12, int(bar_height * 0.85))
    text_renderer = TextRenderer(video_config, font_size=font_size)

    # Create new video with extra height for action bar
    vid_with_bars = processor.create_video_with_bars(video, bar_height)
    processor.add_base_video_content(vid_with_bars, video, bar_height)

    # Total frames for display (excluding repeated frames at end)
    total_frames = (
        total_episode_frames if total_episode_frames is not None else len(actions)
    )

    # Add action indicator bars for each frame
    for t in range(time_steps):
        # Get current action, handling repeated frames
        if t < len(actions):
            current_action = actions[t]
            current_done = dones[t] if dones and t < len(dones) else False
            frame_num = t + 1  # 1-indexed for display
        else:
            # For repeated frames, use the last action
            current_action = actions[-1] if actions else 0
            current_done = dones[-1] if dones else False
            frame_num = total_frames  # Show last frame number for repeated frames

        # Get the bar color based on action (green for normal, red for OOD)
        bar_color = score_renderer.get_bar_color(current_action)

        # Fill the entire bar area with the action color
        vid_with_bars[t, :, :bar_height, :] = np.array(bar_color)[
            :, np.newaxis, np.newaxis
        ]

        # Add text overlay with frame counter, done status, and action status
        done_marker = " [DONE]" if current_done else ""
        action_status = "OOD" if current_action == 1 else "Normal"
        bar_text = f"F{frame_num}/{total_frames}{done_marker} | {action_status}"
        _text_width, text_height = text_renderer.calculate_text_dimensions(bar_text)
        text_x, text_y = text_renderer.calculate_text_position(bar_height, text_height)

        vid_with_bars[t] = text_renderer.add_text_to_frame(
            vid_with_bars[t], bar_text, (text_x, text_y)
        )

    return vid_with_bars


def generate_caption(threshold: float, afhp: float, video_data: Dict[str, Any]) -> str:
    """Generate video caption with relevant information."""
    caption = f"Threshold: {threshold:.2E} - AFHP: {afhp:.2f}"

    use_recons = video_data["reconstructions"][0] is not None

    # Check if human observations are present
    human_obs = video_data.get("human_observations", [])
    has_human_view = human_obs and any(h is not None for h in human_obs)

    if has_human_view:
        if use_recons:
            caption += " - Top: Agent view (Left: Original, Right: Reconstruction), Bottom: Human view (512x512)"
        else:
            caption += " - Top: Agent view (64x64), Bottom: Human view (512x512)"
    else:
        if use_recons:
            caption += " - Left: Original, Right: Reconstruction"
        else:
            caption += " - Original observations"

    if video_data["scores"] is not None and any(
        score is not None for score in video_data["scores"]
    ):
        score_min, score_max = ScoreBarRenderer.calculate_score_bounds_static(
            video_data["scores"]
        )
        caption += (
            " - Score bar: (Green=Normal, Red=OOD, Range: "
            f"{score_min:.3f}-{score_max:.3f})"
        )
    else:
        # When scores are not available, we show action indicator instead
        caption += " - Action bar: (Green=Normal, Red=OOD Detected)"

    return caption


def log_to_wandb(
    logger: WandbLogger,
    video: np.ndarray,
    caption: str,
    afhp: float,
    video_config: dict,
    wandb_category: Optional[str] = None,
) -> None:
    """Log video to WandB with optional category organization."""
    # Create the video key with category prefix if provided
    if wandb_category:
        video_key = f"{wandb_category}/episode_{afhp:.2f}"
    else:
        video_key = f"eval_episode_{afhp:.2f}"

    logger.experiment.log(
        {
            video_key: wandb.Video(
                video,
                fps=video_config["fps"],
                format="gif",
                caption=caption,
            ),
        }
    )


def resolve_video_output_folder(
    output_folder: Optional[str], eval_run_dir: Path, create_folder: bool = True
) -> Optional[Path]:
    """Resolve the video output folder path relative to eval_run_dir."""
    if output_folder is None:
        return None

    output_path = Path(output_folder)

    # If it's an absolute path, use it as-is
    if output_path.is_absolute():
        resolved_path = output_path
    else:
        # If it's a relative path, make it relative to eval_run_dir
        resolved_path = eval_run_dir / output_folder

    # Only create the folder if requested
    if create_folder:
        resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def save_video_to_folder(
    video: np.ndarray,
    folder_path: Path,
    filename: str,
    video_config: dict,
    caption: str = "",
) -> None:
    """
    Save video to a local folder as GIF.

    Args:
        video: Video array in (T, C, H, W) format
        folder_path: Path to the folder where video should be saved
        filename: Base filename (without extension)
        video_config: Video configuration dictionary
        caption: Optional caption for the video
    """
    # Ensure folder exists
    folder_path.mkdir(parents=True, exist_ok=True)

    # Convert video from (T, C, H, W) to list of PIL Images
    t0 = time.perf_counter()
    frames = []
    for t in range(video.shape[0]):
        # Convert from (C, H, W) to (H, W, C)
        frame = video[t].transpose(1, 2, 0)
        # Ensure values are in 0-255 range
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        # Create PIL Image
        pil_frame = Image.fromarray(frame, mode="RGB")
        frames.append(pil_frame)
    frame_convert_time = time.perf_counter() - t0

    # Save as GIF
    if frames:
        output_path = folder_path / f"{filename}.gif"
        t0 = time.perf_counter()
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(
                1000 / video_config["fps"]
            ),  # Convert fps to milliseconds per frame
            loop=0,  # Infinite loop
            optimize=False,  # Keep quality high for debugging
        )
        gif_encode_time = time.perf_counter() - t0

        video_logger.debug(
            f"save_video_to_folder: frame_convert={frame_convert_time:.3f}s, "
            f"gif_encode={gif_encode_time:.3f}s, frames={len(frames)}, "
            f"size={video.shape}"
        )

        # Save caption as text file if provided
        if caption:
            caption_path = folder_path / f"{filename}_caption.txt"
            caption_path.write_text(caption)


def process_and_log_video(
    episode: List[Dict],
    episode_idx: int,
    threshold: float,
    afhp: float,
    video_config: dict,
    output_folder: Optional[Path] = None,
    logger: Optional[WandbLogger] = None,
    logging_mode: Literal["wandb", "folder", "both", "none"] = "wandb",
    subfolder: Optional[str] = None,
    wandb_category: Optional[str] = None,
    skip_score_normalization: bool = False,
    include_human_view: bool = False,
) -> None:
    """
    Complete video processing and logging pipeline.

    Args:
        episode: List of collected state data for an episode
        episode_idx: Index of the episode to process
        logger: WandbLogger instance (required for wandb mode)
        threshold: Threshold value for the video caption
        afhp: Ask for help percentage
        video_config: Video configuration dictionary
        output_folder: Folder path for saving videos (required for folder and both modes)
        logging_mode: Logging mode - "wandb", "folder", "both", or "none"
        subfolder: Optional subfolder name to create within output_folder for organization
        wandb_category: Optional category name for wandb logging organization
        skip_score_normalization: If True, don't normalize scores (useful for max_prob metric)
        include_human_view: If True, include the human-resolution view in the video (default: False)
    """
    video_logger.debug(
        f"[ep={episode_idx}] process_and_log_video called: logging_mode={logging_mode}, "
        f"episode_frames={len(episode)}, logger={'present' if logger is not None else 'None'}"
    )

    # Skip video logging entirely if mode is "none"
    if logging_mode == "none":
        video_logger.debug(f"[ep={episode_idx}] Skipping video (logging_mode='none')")
        return

    total_start = time.perf_counter()
    timings = {}
    num_frames = len(episode)

    video_logger.debug(
        f"[ep={episode_idx}] Starting video processing ({num_frames} frames)..."
    )

    # Extract and validate data
    video_logger.debug(f"[ep={episode_idx}] Step 1/6: Extracting video data...")
    t0 = time.perf_counter()
    video_data = extract_video_data(episode)
    timings["extract_data"] = time.perf_counter() - t0
    video_logger.debug(
        f"[ep={episode_idx}] Step 1/6: Done ({timings['extract_data']:.3f}s)"
    )

    # Create video processor instance
    processor = VideoProcessor(video_config)

    # Process video frames (agent observations with optional reconstructions)
    video_logger.debug(
        f"[ep={episode_idx}] Step 2/6: Combining observations and reconstructions..."
    )
    t0 = time.perf_counter()
    combined_video = processor.combine_observations_and_reconstructions(
        video_data["observations"],
        video_data["reconstructions"],
    )
    timings["combine_obs_recons"] = time.perf_counter() - t0
    video_logger.debug(
        f"[ep={episode_idx}] Step 2/6: Done ({timings['combine_obs_recons']:.3f}s), shape={combined_video.shape}"
    )

    video_logger.debug(f"[ep={episode_idx}] Step 3/6: Adding repeated frames...")
    t0 = time.perf_counter()
    agent_video = processor.add_repeated_frames(combined_video)
    timings["add_repeated_frames"] = time.perf_counter() - t0
    video_logger.debug(
        f"[ep={episode_idx}] Step 3/6: Done ({timings['add_repeated_frames']:.3f}s)"
    )

    # Check if we have human view
    human_obs = video_data.get("human_observations", [])
    has_human_view = (
        include_human_view and human_obs and any(h is not None for h in human_obs)
    )

    # Store total episode frames for display (before repeated frames were added)
    total_episode_frames = num_frames

    score_renderer = ScoreBarRenderer(video_config)
    has_scores = video_data["scores"] is not None and any(
        score is not None for score in video_data["scores"]
    )

    if has_human_view:
        # Step 4: Process agent and human views separately, adding bars to each
        video_logger.debug(
            f"[ep={episode_idx}] Step 4/6: Processing agent and human views with separate bars..."
        )
        t0 = time.perf_counter()

        # Get human frame dimensions to scale agent video appropriately
        human_height, human_width = None, None
        for h in human_obs:
            if h is not None:
                human_height, human_width = h.shape[0], h.shape[1]
                break

        # Upscale agent video to match human width
        agent_video_scaled = processor.upscale_video(agent_video, human_width)

        # Add bars to agent view
        if has_scores:
            agent_with_bars = add_score_bars(
                agent_video_scaled,
                video_data["scores"],
                video_data["actions"],
                score_renderer,
                video_config,
                skip_normalization=skip_score_normalization,
                total_episode_frames=total_episode_frames,
                dones=video_data["dones"],
            )
        else:
            agent_with_bars = add_action_indicator_bars(
                agent_video_scaled,
                video_data["actions"],
                score_renderer,
                video_config,
                total_episode_frames=total_episode_frames,
                dones=video_data["dones"],
            )

        # Process human observations into video array
        time_steps = agent_video.shape[0]
        human_video = np.zeros(
            (time_steps, 3, human_height, human_width), dtype=np.uint8
        )
        for t in range(time_steps):
            # Use the corresponding human frame, or last available for repeated frames
            if t < len(human_obs) and human_obs[t] is not None:
                human_video[t] = human_obs[t].transpose(2, 0, 1)
            elif t > 0:
                # Copy previous frame for repeated frames or missing frames
                idx = min(t, len(human_obs) - 1)
                while idx >= 0 and human_obs[idx] is None:
                    idx -= 1
                if idx >= 0 and human_obs[idx] is not None:
                    human_video[t] = human_obs[idx].transpose(2, 0, 1)
                else:
                    human_video[t] = 128  # Gray fill if no frames available
            else:
                human_video[t] = 128  # Gray fill

        # Add bars to human view
        if has_scores:
            human_with_bars = add_score_bars(
                human_video,
                video_data["scores"],
                video_data["actions"],
                score_renderer,
                video_config,
                skip_normalization=skip_score_normalization,
                total_episode_frames=total_episode_frames,
                dones=video_data["dones"],
            )
        else:
            human_with_bars = add_action_indicator_bars(
                human_video,
                video_data["actions"],
                score_renderer,
                video_config,
                total_episode_frames=total_episode_frames,
                dones=video_data["dones"],
            )

        # Stack agent (with bars) on top of human (with bars)
        combined_video = np.concatenate([agent_with_bars, human_with_bars], axis=2)

        timings["process_views_with_bars"] = time.perf_counter() - t0
        video_logger.debug(
            f"[ep={episode_idx}] Step 4/6: Done ({timings['process_views_with_bars']:.3f}s), shape={combined_video.shape}"
        )

        # Step 5: Skip (bars already added)
        video_logger.debug(
            f"[ep={episode_idx}] Step 5/6: Skipped (bars added in step 4)"
        )
    else:
        # No human view - upscale agent video to minimum output size for readability
        video_logger.debug(f"[ep={episode_idx}] Step 4/6: Upscaling agent view...")
        t0 = time.perf_counter()
        min_size = video_config.get("min_output_size", 512)
        combined_video = processor.upscale_video(agent_video, min_size)
        timings["upscale_video"] = time.perf_counter() - t0
        video_logger.debug(
            f"[ep={episode_idx}] Step 4/6: Done ({timings['upscale_video']:.3f}s), shape={combined_video.shape}"
        )

        # Step 5: Add bars at the final resolution
        video_logger.debug(f"[ep={episode_idx}] Step 5/6: Adding score bars...")
        t0 = time.perf_counter()
        if has_scores:
            combined_video = add_score_bars(
                combined_video,
                video_data["scores"],
                video_data["actions"],
                score_renderer,
                video_config,
                skip_normalization=skip_score_normalization,
                total_episode_frames=total_episode_frames,
                dones=video_data["dones"],
            )
        else:
            combined_video = add_action_indicator_bars(
                combined_video,
                video_data["actions"],
                score_renderer,
                video_config,
                total_episode_frames=total_episode_frames,
                dones=video_data["dones"],
            )
        timings["add_score_bars"] = time.perf_counter() - t0
        video_logger.debug(
            f"[ep={episode_idx}] Step 5/6: Done ({timings['add_score_bars']:.3f}s)"
        )

    # Generate caption
    caption = generate_caption(threshold, afhp, video_data)
    # Include subfolder/category info in filename if provided
    if subfolder or wandb_category:
        category_suffix = f"_{subfolder or wandb_category}"
        filename = f"episode_{episode_idx}_afhp_{afhp:.2f}{category_suffix}"
    else:
        filename = f"episode_{episode_idx}_afhp_{afhp:.2f}"

    # Log based on mode
    if logging_mode in ["wandb", "both"]:
        if logger is None:
            raise ValueError("logger is required for wandb logging mode")
        video_logger.debug(f"[ep={episode_idx}] Step 6/6: Uploading to wandb...")
        t0 = time.perf_counter()
        log_to_wandb(
            logger, combined_video, caption, afhp, video_config, wandb_category
        )
        timings["log_to_wandb"] = time.perf_counter() - t0
        video_logger.debug(
            f"[ep={episode_idx}] Step 6/6: wandb upload done ({timings['log_to_wandb']:.3f}s)"
        )

    if logging_mode in ["folder", "both"]:
        if output_folder is None:
            raise ValueError(
                "output_folder is required for folder and both logging modes"
            )
        # Create subfolder if specified
        target_folder = output_folder / subfolder if subfolder else output_folder
        video_logger.debug(
            f"[ep={episode_idx}] Step 6/6: Saving to folder {target_folder}..."
        )
        t0 = time.perf_counter()
        save_video_to_folder(
            combined_video, target_folder, filename, video_config, caption
        )
        timings["save_to_folder"] = time.perf_counter() - t0
        video_logger.debug(
            f"[ep={episode_idx}] Step 6/6: Folder save done ({timings['save_to_folder']:.3f}s)"
        )

    total_time = time.perf_counter() - total_start
    timings["total"] = total_time

    # Log profiling summary
    video_shape = combined_video.shape
    timing_str = ", ".join(f"{k}={v:.3f}s" for k, v in timings.items())
    video_logger.info(
        f"Video profiling (ep={episode_idx}, frames={num_frames}, shape={video_shape}): {timing_str}"
    )

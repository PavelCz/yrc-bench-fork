"""
Consolidated video utilities for evaluation video generation.

This module contains classes and functions for:
- Processing video frames and combining observations with reconstructions
- Rendering score bars and text overlays
- Extracting episode video data and logging videos to WandB
"""

import logging
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Literal
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image, ImageDraw, ImageFont


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

    def add_repeated_frames(self, video: np.ndarray) -> np.ndarray:
        """
        Add repeated frames at the end for smoother video ending.
        
        Args:
            video: Input video array
            
        Returns:
            Video with repeated final frames appended
        """
        final_frame = video[-1:].copy()
        repeated_frames = np.repeat(final_frame, self.config['final_frame_repetitions'], axis=0)
        return np.concatenate([video, repeated_frames], axis=0)

    def create_video_with_bars(self, video: np.ndarray) -> np.ndarray:
        """
        Create a new video array with space for score bars.
        
        Args:
            video: Input video array
            
        Returns:
            New video array with extra height for score bars
        """
        time_steps, channels, height, width = video.shape
        bar_height = self.config['score_bar_height']

        return np.zeros(
            (time_steps, channels, height + bar_height, width), dtype=np.uint8
        )

    def add_base_video_content(self, video_with_bars: np.ndarray, original_video: np.ndarray) -> None:
        """
        Copy original video content below the bar area.
        
        Args:
            video_with_bars: Video array with space for bars
            original_video: Original video content to copy
        """
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

        # If all scores are -inf, return 0.0, 1.0
        if len(all_rational_scores) == 0:
            return 0.0, 1.0
        score_min = np.min(all_rational_scores)
        score_max = np.max(all_scores)
        return score_min, score_max

    def normalize_score(self, score: float, score_min: float, score_max: float) -> float:
        """Normalize a score to 0-1 range."""
        if score_max > score_min:
            normalized_score = (score - score_min) / (score_max - score_min)
            return float(np.clip(normalized_score, 0, 1))
        return 0.5

    def get_bar_color(self, action: int) -> List[int]:
        """Get the appropriate color for the score bar based on action."""
        if action == 1:  # OOD detected
            return self.config['ood_color']
        return self.config['normal_color']

    def calculate_bar_dimensions(self, normalized_score: float, width: int) -> Tuple[int, bool]:
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

    def __init__(self, config: dict):
        self.config = config
        self.font = self._load_font()

    def _load_font(self) -> Optional[ImageFont.FreeTypeFont]:
        try:
            return ImageFont.truetype("arial.ttf", self.config['font_size'])
        except OSError:
            try:
                return ImageFont.load_default()
            except OSError:
                return None

    def calculate_text_dimensions(self, text: str) -> Tuple[int, int]:
        if self.font:
            bbox = self.font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        return len(text) * self.config['char_width_estimate'], self.config['font_size'] + 2

    def calculate_text_position(self, bar_height: int, text_height: int) -> Tuple[int, int]:
        text_x = self.config['text_padding']
        text_y = (bar_height - text_height) // 2
        return text_x, text_y

    def add_text_to_frame(self, frame: np.ndarray, text: str, position: Tuple[int, int]) -> np.ndarray:
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
            # Draw main text
            draw.text((text_x, text_y), text, fill=tuple(self.config['text_color']), font=self.font)
        else:
            draw.text((text_x, text_y), text, fill=tuple(self.config['text_color']))

        # Convert back to (C, H, W)
        return np.array(pil_image).transpose(2, 0, 1)


def extract_video_data(collected_states: List[List[Dict]], episode_idx: int) -> Optional[Dict[str, Any]]:
    """Extract and validate video data from collected states."""
    if episode_idx >= len(collected_states) or not collected_states[episode_idx]:
        logging.warning(f"No data available for episode {episode_idx}")
        return None

    episode_data = collected_states[episode_idx]
    obs = [x["obs"] for x in episode_data]
    scores = [x["scores"] for x in episode_data]
    recons = [x["recons"] for x in episode_data]
    actions = [x["action"] for x in episode_data]

    return {
        'observations': obs,
        'scores': scores,
        'reconstructions': recons,
        'actions': actions,
    }


def add_score_bars(
    video: np.ndarray,
    scores: List[float],
    actions: List[int],
    score_renderer: ScoreBarRenderer,
    text_renderer: TextRenderer,
    video_config: dict,
) -> np.ndarray:
    """Add score bars to video frames."""
    score_min, score_max = score_renderer.calculate_score_bounds(scores)
    bar_height = video_config['score_bar_height']
    time_steps, channels, _, width = video.shape

    # Create new video with extra height for score bar
    processor = VideoProcessor(video_config)
    vid_with_bars = processor.create_video_with_bars(video)
    processor.add_base_video_content(vid_with_bars, video)

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
            vid_with_bars[t, :, :bar_height, bar_width:] = video_config['score_bar_bg_color']

        # Add text overlay with score value
        score_text = f"{current_score:.3f}"
        _text_width, text_height = text_renderer.calculate_text_dimensions(score_text)
        text_x, text_y = text_renderer.calculate_text_position(bar_height, text_height)

        vid_with_bars[t] = text_renderer.add_text_to_frame(vid_with_bars[t], score_text, (text_x, text_y))

    return vid_with_bars


def generate_caption(threshold: float, afhp: float, video_data: Dict[str, Any]) -> str:
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


def log_to_wandb(logger: WandbLogger, video: np.ndarray, caption: str, afhp: float, video_config: dict) -> None:
    """Log video to WandB."""
    logger.experiment.log({
        f"eval_episode_{afhp:.2f}": wandb.Video(
            video,
            fps=video_config['fps'],
            format="gif",
            caption=caption,
        ),
    })


def save_video_to_folder(
    video: np.ndarray,
    folder_path: str,
    filename: str,
    video_config: dict,
    caption: str = ""
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
    os.makedirs(folder_path, exist_ok=True)

    # Convert video from (T, C, H, W) to list of PIL Images
    frames = []
    for t in range(video.shape[0]):
        # Convert from (C, H, W) to (H, W, C)
        frame = video[t].transpose(1, 2, 0)
        # Ensure values are in 0-255 range
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        # Create PIL Image
        pil_frame = Image.fromarray(frame, mode="RGB")
        frames.append(pil_frame)

    # Save as GIF
    if frames:
        output_path = os.path.join(folder_path, f"{filename}.gif")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / video_config['fps']),  # Convert fps to milliseconds per frame
            loop=0,  # Infinite loop
            optimize=False  # Keep quality high for debugging
        )

        # Save caption as text file if provided
        if caption:
            caption_path = os.path.join(folder_path, f"{filename}_caption.txt")
            with open(caption_path, 'w') as f:
                f.write(caption)


def process_and_log_video(
    collected_states: List[List[Dict]],
    episode_idx: int,
    logger: Optional[WandbLogger] = None,
    threshold: float = 0.0,
    afhp: float = 0.0,
    video_config: dict = None,
    output_folder: Optional[str] = None,
    logging_mode: Literal["wandb", "folder", "both"] = "wandb",
) -> None:
    """
    Complete video processing and logging pipeline.

    Args:
        collected_states: List of collected state data for all episodes
        episode_idx: Index of the episode to process
        logger: WandbLogger instance (required for wandb mode)
        threshold: Threshold value for the video caption
        afhp: Ask for help percentage
        video_config: Video configuration dictionary
        output_folder: Folder path for saving videos (required for folder mode)
        logging_mode: Logging mode - "wandb", "folder", or "both"
    """
    if video_config is None:
        raise ValueError("video_config is required")

    # Extract and validate data
    video_data = extract_video_data(collected_states, episode_idx)
    if not video_data:
        return

    # Create video processor instance
    processor = VideoProcessor(video_config)

    # Process video frames
    combined_video = processor.combine_observations_and_reconstructions(
        video_data['observations'],
        video_data['reconstructions'],
    )

    combined_video = processor.add_repeated_frames(combined_video)

    # Add score bars if available
    if video_data['scores'] is not None:
        score_renderer = ScoreBarRenderer(video_config)
        text_renderer = TextRenderer(video_config)

        combined_video = add_score_bars(
            combined_video,
            video_data['scores'],
            video_data['actions'],
            score_renderer,
            text_renderer,
            video_config,
        )

    # Generate caption
    caption = generate_caption(threshold, afhp, video_data)
    filename = f"episode_{episode_idx}_afhp_{afhp:.2f}"

    # Log based on mode
    if logging_mode in ["wandb", "both"]:
        if logger is None:
            raise ValueError("logger is required for wandb logging mode")
        log_to_wandb(logger, combined_video, caption, afhp, video_config)

    if logging_mode in ["folder", "both"]:
        if output_folder is None:
            raise ValueError("output_folder is required for folder logging mode")
        save_video_to_folder(combined_video, output_folder, filename, video_config, caption)


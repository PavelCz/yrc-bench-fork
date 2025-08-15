#!/usr/bin/env python3
"""
Coinrun Counterfactual Analysis Script

This script performs counterfactual analysis for the coinrun environment by:
1. Rolling out the weak agent in coinrun with randomly placed coin
2. Testing different level seeds to find failure cases
3. Running the same level seed with deterministic coin placement (random_percent=0)
4. Saving rollouts as videos for analysis
5. Logging level seeds and results for reference

Usage:
    python coinrun_counterfactual_analysis.py --weak_agent_path <path> --output_dir <dir> [options]
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime

# Add YRC to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
import torch
import imageio

from YRC.core.configs.global_configs import set_global_variable
from YRC.envs.procgen import load_policy
from lib.procgenAISC.procgen import ProcgenEnv
from YRC.envs.procgen.wrappers import (
    VecExtractDictObs,
    TransposeFrame,
    ScaledFloatFrame,
    HardResetWrapper,
)


class CoinrunCounterfactualAnalyzer:
    """Performs counterfactual analysis for coinrun environment."""

    def __init__(self, weak_agent_path: str, output_dir: str, device: str = "cpu", scale: int = 1):
        """
        Initialize the analyzer.

        Args:
            weak_agent_path: Path to the weak agent checkpoint
            output_dir: Directory to save results and videos
            device: Device to run models on
        """
        # Validate inputs
        if not os.path.exists(weak_agent_path):
            raise FileNotFoundError(
                f"Weak agent checkpoint not found: {weak_agent_path}"
            )

        self.weak_agent_path = weak_agent_path
        self.output_dir = output_dir
        self.device = device
        self.scale = int(scale) if isinstance(scale, int) and scale >= 1 else 1

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize global variables for YRC
        set_global_variable("device", torch.device(device))
        set_global_variable("benchmark", "procgen")

        self.logger.info("Initialized CoinrunCounterfactualAnalyzer")
        self.logger.info(f"Weak agent: {weak_agent_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Video scale: {self.scale}x (pixel-perfect)")

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "coinrun_analysis.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def create_env(
        self, random_percent: int = 100, start_level: int = 0, num_levels: int = 1
    ):
        """
        Create a coinrun environment with specified parameters.

        Args:
            random_percent: Percentage of coin randomization (0=deterministic, 100=fully random)
            start_level: Starting level seed
            num_levels: Number of levels to include

        Returns:
            Wrapped procgen environment
        """
        # Create base environment
        env = ProcgenEnv(
            env_name="coinrun",
            num_envs=1,
            num_threads=1,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode="hard",
            rand_seed=start_level,  # Use level as seed for consistency
            use_backgrounds=True,
            use_monochrome_assets=False,
            restrict_themes=False,
            random_percent=random_percent,  # Key parameter for counterfactual analysis
        )

        # Apply wrappers (same as in YRC framework)
        env = VecExtractDictObs(env, "rgb")
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
        env = HardResetWrapper(env)
        env.obs_shape = env.observation_space.shape

        return env

    def load_weak_agent(self, env):
        """Load the weak agent policy."""
        self.logger.info(f"Loading weak agent from {self.weak_agent_path}")
        try:
            agent = load_policy(self.weak_agent_path, env)
            agent.eval()  # Set to evaluation mode
            self.logger.info("Weak agent loaded successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to load weak agent: {e}")
            raise

    def rollout_episode(
        self, agent, env, max_steps: int = 1000, record_video: bool = True
    ):
        """
        Roll out a single episode with the agent.

        Args:
            agent: The policy to roll out
            env: Environment to run in
            max_steps: Maximum steps per episode
            record_video: Whether to record video frames

        Returns:
            Tuple of (total_reward, episode_length, frames, success, invisible_coin_collected)
        """
        obs = env.reset()
        total_reward = 0.0
        episode_length = 0
        frames = []
        done_flag = False
        invisible_coin_collected_any = False

        # Record initial frame
        if record_video:
            frame = self.obs_to_frame(obs)
            if frame is not None:
                frames.append(frame)

        while not done_flag and episode_length < max_steps:
            # Get action from agent
            action = agent.act(obs, greedy=True)

            # Step environment
            obs, reward, done, info = env.step(action)

            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1

            info_item = info[0] if isinstance(info, list) and len(info) > 0 else {}

            if "invisible_coin_collected" in info_item:
                if info_item["invisible_coin_collected"] == 1:
                    invisible_coin_collected_any = True
            else:
                raise ValueError(
                    f"invisible_coin_collected not found in info: {info_item}"
                )

            # Record frame
            if record_video:
                frame = self.obs_to_frame(obs)
                if frame is not None:
                    frames.append(frame)

            done_flag = bool(done[0]) if isinstance(done, np.ndarray) else bool(done)
            if done_flag:
                break

        # Success is typically indicated by reaching the goal (positive reward)
        success = total_reward > 0

        return total_reward, episode_length, frames, success, invisible_coin_collected_any

    def obs_to_frame(self, obs):
        """Convert current observation to an RGB frame (H,W,3) uint8."""
        try:
            arr = obs
            if isinstance(arr, dict):
                if "rgb" in arr:
                    arr = arr["rgb"]
                elif "image" in arr:
                    arr = arr["image"]
                else:
                    return None
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            elif hasattr(arr, "numpy"):
                arr = arr.numpy()

            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3:
                # If channel-first, transpose to HWC
                if arr.shape[0] in (1, 3):
                    if arr.shape[0] == 1:
                        arr = np.repeat(arr, 3, axis=0)
                    arr = np.transpose(arr, (1, 2, 0))
                # else assume already HWC
            else:
                return None

            # Scale to uint8 if needed
            if arr.dtype != np.uint8:
                # Assume values in [0,1]
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            return arr
        except Exception:
            return None

    def save_video(self, frames, filename: str, fps: int = 30):
        """Save frames as a video file."""
        if not frames:
            self.logger.warning(f"No frames to save for {filename}")
            return

        try:
            # Validate frame shape
            first = frames[0]
            if first.ndim != 3 or first.shape[2] != 3:
                self.logger.error(
                    f"Unexpected frame shape {first.shape} for {filename}"
                )
                return

            # Pixel-perfect integer upscaling via nearest-neighbor
            scale = max(1, int(self.scale))
            if scale > 1:
                def upscale(img):
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
                frames_to_write = [upscale(f) for f in frames]
            else:
                frames_to_write = [f if f.dtype == np.uint8 else np.clip(f, 0, 255).astype(np.uint8) for f in frames]

            video_path = os.path.join(self.output_dir, filename)
            with imageio.get_writer(video_path, fps=fps, codec="libx264") as w:
                for frame in frames_to_write:
                    w.append_data(frame)
            self.logger.info(f"Video saved: {video_path}")

        except Exception as e:
            self.logger.error(f"Failed to save video {filename}: {e}")

    def find_failure_seeds(self, desired_count: int, max_attempts: int = 100, start_seed: int = 0):
        """
        Find multiple level seeds where the weak agent fails (gets 0 reward) with random coin placement
        and the invisible coin was not collected.

        Args:
            desired_count: Target number of failure seeds to find
            max_attempts: Maximum number of seeds to try (upper bound on seeds tested)
            start_seed: Starting seed value

        Returns:
            List of level seeds (length <= desired_count)
        """
        self.logger.info(
            f"Searching for up to {desired_count} failure seeds (max {max_attempts} attempts)..."
        )

        found: list[int] = []
        attempt = 0

        while attempt < max_attempts and len(found) < desired_count:
            seed = start_seed + attempt
            attempt += 1

            # Create environment with random coin placement
            env = self.create_env(random_percent=100, start_level=seed, num_levels=1)
            agent = self.load_weak_agent(env)

            # Test rollout
            reward, length, _, success, invisible_coin_collected = self.rollout_episode(
                agent, env, record_video=False
            )

            self.logger.info(
                f"Seed {seed}: reward={reward:.2f}, length={length}, success={success}, invisible_coin_collected={invisible_coin_collected}"
            )

            # Clean up
            env.close()

            # If agent failed (no success) and invisible coin was NOT collected, record the seed
            if (not success) and (not invisible_coin_collected):
                self.logger.info(f"Found failure seed: {seed}")
                found.append(seed)

        if len(found) == 0:
            self.logger.warning(f"No failure seed found in {max_attempts} attempts")
        else:
            self.logger.info(f"Found {len(found)} failure seeds: {found}")

        return found

    def run_counterfactual_analysis(self, failure_seed: int):
        """
        Run counterfactual analysis on a specific seed.

        Args:
            failure_seed: The level seed to analyze

        Returns:
            Dictionary with analysis results
        """
        self.logger.info(f"Running counterfactual analysis for seed {failure_seed}")

        results = {
            "seed": failure_seed,
            "timestamp": datetime.now().isoformat(),
            "random_placement": {},
            "deterministic_placement": {},
        }

        # Test 1: Random coin placement (random_percent=100)
        self.logger.info("Testing with random coin placement (random_percent=100)...")
        env_random = self.create_env(
            random_percent=100, start_level=failure_seed, num_levels=1
        )
        agent_random = self.load_weak_agent(env_random)

        reward_random, length_random, frames_random, success_random, invisible_random = (
            self.rollout_episode(agent_random, env_random, record_video=True)
        )

        results["random_placement"] = {
            "reward": float(reward_random),
            "episode_length": int(length_random),
            "success": bool(success_random),
            "num_frames": len(frames_random),
            "invisible_coin_collected": bool(invisible_random),
        }

        # Save video for random placement
        self.save_video(frames_random, f"seed_{failure_seed}_random_placement.mp4")
        env_random.close()

        # Test 2: Deterministic coin placement (random_percent=0)
        self.logger.info(
            "Testing with deterministic coin placement (random_percent=0)..."
        )
        env_deterministic = self.create_env(
            random_percent=0, start_level=failure_seed, num_levels=1
        )
        agent_deterministic = self.load_weak_agent(env_deterministic)

        (
            reward_deterministic,
            length_deterministic,
            frames_deterministic,
            success_deterministic,
            invisible_deterministic,
        ) = self.rollout_episode(
            agent_deterministic, env_deterministic, record_video=True
        )

        results["deterministic_placement"] = {
            "reward": float(reward_deterministic),
            "episode_length": int(length_deterministic),
            "success": bool(success_deterministic),
            "num_frames": len(frames_deterministic),
            "invisible_coin_collected": bool(invisible_deterministic),
        }

        # Save video for deterministic placement
        self.save_video(
            frames_deterministic, f"seed_{failure_seed}_deterministic_placement.mp4"
        )
        env_deterministic.close()

        # Log comparison
        self.logger.info("=== Counterfactual Analysis Results ===")
        self.logger.info(f"Seed: {failure_seed}")
        self.logger.info(
            f"Random placement (100%): reward={reward_random:.2f}, "
            f"length={length_random}, success={success_random}"
        )
        self.logger.info(
            f"Deterministic placement (0%): reward={reward_deterministic:.2f}, "
            f"length={length_deterministic}, success={success_deterministic}"
        )

        return results

    def run_analysis(self, max_seed_attempts: int = 100, start_seed: int = 0, num_failures: int = 1):
        """
        Run the complete counterfactual analysis pipeline.

        Args:
            max_seed_attempts: Maximum seeds to try when finding failure case
            start_seed: Starting seed value

        Returns:
            Dictionary with complete analysis results
        """
        self.logger.info("Starting coinrun counterfactual analysis")

        # Step 1: Find failure seeds as requested
        seeds = self.find_failure_seeds(num_failures, max_seed_attempts, start_seed)

        if len(seeds) == 0:
            self.logger.error(
                "Could not find any failure seed. Analysis cannot continue."
            )
            return {"error": "No failure seed found", "seeds": []}

        # Step 2: Run counterfactual analysis on each seed
        all_results = {}
        for seed in seeds:
            res = self.run_counterfactual_analysis(seed)
            all_results[seed] = res

            # Save per-seed results
            results_file = os.path.join(
                self.output_dir, f"analysis_results_seed_{seed}.json"
            )
            with open(results_file, "w") as f:
                json.dump(res, f, indent=2)

        # Step 3: Save summary
        summary_file = os.path.join(self.output_dir, "analysis_results_summary.json")
        with open(summary_file, "w") as f:
            json.dump({"seeds": seeds, "results": all_results}, f, indent=2)

        self.logger.info(
            f"Analysis complete. Results saved for seeds {seeds}. Summary: {summary_file}"
        )
        return {"seeds": seeds, "results": all_results}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Coinrun Counterfactual Analysis")
    parser.add_argument(
        "--weak_agent_path",
        default="YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth",
        help="Path to weak agent checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/coinrun_analysis_output",
        help="Directory to save results and videos",
    )
    parser.add_argument(
        "--max_seed_attempts",
        type=int,
        default=100,
        help="Maximum number of seeds to try when finding failure case",
    )
    parser.add_argument(
        "--num_failures",
        type=int,
        default=1,
        help="Number of failure seeds to find before stopping",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Integer pixel-perfect upscaling factor for saved videos",
    )
    parser.add_argument(
        "--start_seed", type=int, default=0, help="Starting seed value for search"
    )
    args = parser.parse_args()

    # Check if weak agent checkpoint exists
    if not os.path.exists(args.weak_agent_path):
        print(f"Error: Weak agent checkpoint not found at {args.weak_agent_path}")
        print(
            "Please ensure the checkpoint file exists or provide correct path with --weak_agent_path"
        )
        sys.exit(1)

    # Resolve device automatically if requested
    device = "cuda" if ("torch" in globals() and torch.cuda.is_available()) else "cpu"

    # Create analyzer and run analysis
    try:
        analyzer = CoinrunCounterfactualAnalyzer(
            weak_agent_path=args.weak_agent_path,
            output_dir=args.output_dir,
            device=device,
            scale=args.scale,
        )

        results = analyzer.run_analysis(
            max_seed_attempts=args.max_seed_attempts,
            start_seed=args.start_seed,
            num_failures=args.num_failures,
        )

        if "error" not in results:
            print("\n=== Analysis Complete ===")
            print(f"Results saved in: {args.output_dir}")
            print(f"Failure seeds found: {results['seeds']}")
            print("Videos saved for both random and deterministic coin placement for each seed")
        else:
            print(f"Analysis failed: {results['error']}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

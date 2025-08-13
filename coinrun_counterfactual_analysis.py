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
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add YRC to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

# Import dependencies with error handling
def check_and_import_dependencies():
    """Check and import required dependencies with helpful error messages."""
    missing = []
    imports = {}
    
    try:
        import numpy as np
        imports['numpy'] = np
    except ImportError:
        missing.append("numpy")
    
    try:
        import torch
        imports['torch'] = torch
    except ImportError:
        missing.append("torch")
    
    try:
        import cv2
        imports['cv2'] = cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        from YRC.core.configs.global_configs import set_global_variable
        from YRC.envs.procgen import load_policy
        imports['yrc'] = True
    except ImportError as e:
        missing.append(f"YRC framework ({e})")
    
    try:
        from lib.procgenAISC.procgen import ProcgenEnv
        from YRC.envs.procgen.wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame, HardResetWrapper
        imports['procgen'] = True
    except ImportError as e:
        missing.append(f"procgen environment ({e})")
    
    return missing, imports

# Check dependencies first
missing_deps, deps = check_and_import_dependencies()

# Only proceed with imports if we're not just asking for help
if len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help'] and missing_deps:
    print("❌ Missing required dependencies:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\n📋 Installation instructions:")
    print("pip install numpy torch opencv-python gym3")
    print("pip install -e lib/procgenAISC")
    print("\n📖 See README_coinrun_analysis.md for detailed setup instructions")
    print("\n💡 Use --check_deps_only to test dependency installation")
    sys.exit(1)

# Import dependencies if available
if not missing_deps:
    np = deps['numpy']
    torch = deps['torch'] 
    cv2 = deps['cv2']

    from YRC.core.configs.global_configs import set_global_variable
    from YRC.envs.procgen import load_policy
    from lib.procgenAISC.procgen import ProcgenEnv
    from YRC.envs.procgen.wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame, HardResetWrapper


class CoinrunCounterfactualAnalyzer:
    """Performs counterfactual analysis for coinrun environment."""
    
    def __init__(self, weak_agent_path: str, output_dir: str, device: str = "cpu"):
        """
        Initialize the analyzer.
        
        Args:
            weak_agent_path: Path to the weak agent checkpoint
            output_dir: Directory to save results and videos
            device: Device to run models on
        """
        # Validate inputs
        if not os.path.exists(weak_agent_path):
            raise FileNotFoundError(f"Weak agent checkpoint not found: {weak_agent_path}")
        
        self.weak_agent_path = weak_agent_path
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize global variables for YRC
        set_global_variable("device", torch.device(device))
        set_global_variable("benchmark", "procgen")
        
        self.logger.info(f"Initialized CoinrunCounterfactualAnalyzer")
        self.logger.info(f"Weak agent: {weak_agent_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Device: {device}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "coinrun_analysis.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_env(self, random_percent: int = 100, start_level: int = 0, num_levels: int = 1):
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
    
    def rollout_episode(self, agent, env, max_steps: int = 1000, 
                       record_video: bool = True):
        """
        Roll out a single episode with the agent.
        
        Args:
            agent: The policy to roll out
            env: Environment to run in
            max_steps: Maximum steps per episode
            record_video: Whether to record video frames
            
        Returns:
            Tuple of (total_reward, episode_length, frames, success)
        """
        obs = env.reset()
        total_reward = 0.0
        episode_length = 0
        frames = []
        done = False
        
        # Record initial frame
        if record_video:
            # Get RGB frame for video - need to get it from the original environment
            frame = self.get_rgb_frame(env)
            if frame is not None:
                frames.append(frame)
        
        while not done and episode_length < max_steps:
            # Get action from agent
            action = agent.act(obs, greedy=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            
            # Record frame
            if record_video:
                frame = self.get_rgb_frame(env)
                if frame is not None:
                    frames.append(frame)
            
            if done:
                break
        
        # Success is typically indicated by reaching the goal (positive reward)
        success = total_reward > 0
        
        return total_reward, episode_length, frames, success
    
    def get_rgb_frame(self, env):
        """Extract RGB frame from environment for video recording."""
        try:
            # Try to get RGB observation from the base environment
            if hasattr(env, 'venv') and hasattr(env.venv, 'venv'):
                # Unwrap to get to base ProcgenEnv
                base_env = env.venv.venv
                if hasattr(base_env, 'get_images'):
                    images = base_env.get_images()
                    if images is not None and len(images) > 0:
                        return images[0]  # Get first environment's image
            
            # Fallback: try to render
            if hasattr(env, 'render'):
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    return frame
            
            return None
        except Exception as e:
            # Don't log every frame error to avoid spam
            return None
    
    def save_video(self, frames, filename: str, fps: int = 30):
        """Save frames as a video file."""
        if not frames:
            self.logger.warning(f"No frames to save for {filename}")
            return
        
        try:
            # Get dimensions from first frame
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.output_dir, filename)
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
            
            writer.release()
            self.logger.info(f"Video saved: {video_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save video {filename}: {e}")
    
    def find_failure_seed(self, max_attempts: int = 100, start_seed: int = 0):
        """
        Find a level seed where the weak agent fails (gets 0 reward) with random coin placement.
        
        Args:
            max_attempts: Maximum number of seeds to try
            start_seed: Starting seed value
            
        Returns:
            Level seed where agent fails, or None if not found
        """
        self.logger.info(f"Searching for failure seed (max {max_attempts} attempts)...")
        
        for attempt in range(max_attempts):
            seed = start_seed + attempt
            
            # Create environment with random coin placement
            env = self.create_env(random_percent=100, start_level=seed, num_levels=1)
            agent = self.load_weak_agent(env)
            
            # Test rollout
            reward, length, _, success = self.rollout_episode(agent, env, record_video=False)
            
            self.logger.info(f"Seed {seed}: reward={reward:.2f}, length={length}, success={success}")
            
            # Clean up
            env.close()
            
            # If agent failed (no success), we found our seed
            if not success:
                self.logger.info(f"Found failure seed: {seed}")
                return seed
        
        self.logger.warning(f"No failure seed found in {max_attempts} attempts")
        return None
    
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
            "deterministic_placement": {}
        }
        
        # Test 1: Random coin placement (random_percent=100)
        self.logger.info("Testing with random coin placement (random_percent=100)...")
        env_random = self.create_env(random_percent=100, start_level=failure_seed, num_levels=1)
        agent_random = self.load_weak_agent(env_random)
        
        reward_random, length_random, frames_random, success_random = self.rollout_episode(
            agent_random, env_random, record_video=True
        )
        
        results["random_placement"] = {
            "reward": float(reward_random),
            "episode_length": int(length_random),
            "success": bool(success_random),
            "num_frames": len(frames_random)
        }
        
        # Save video for random placement
        self.save_video(frames_random, f"seed_{failure_seed}_random_placement.mp4")
        env_random.close()
        
        # Test 2: Deterministic coin placement (random_percent=0)
        self.logger.info("Testing with deterministic coin placement (random_percent=0)...")
        env_deterministic = self.create_env(random_percent=0, start_level=failure_seed, num_levels=1)
        agent_deterministic = self.load_weak_agent(env_deterministic)
        
        reward_deterministic, length_deterministic, frames_deterministic, success_deterministic = self.rollout_episode(
            agent_deterministic, env_deterministic, record_video=True
        )
        
        results["deterministic_placement"] = {
            "reward": float(reward_deterministic),
            "episode_length": int(length_deterministic),
            "success": bool(success_deterministic),
            "num_frames": len(frames_deterministic)
        }
        
        # Save video for deterministic placement
        self.save_video(frames_deterministic, f"seed_{failure_seed}_deterministic_placement.mp4")
        env_deterministic.close()
        
        # Log comparison
        self.logger.info("=== Counterfactual Analysis Results ===")
        self.logger.info(f"Seed: {failure_seed}")
        self.logger.info(f"Random placement (100%): reward={reward_random:.2f}, "
                        f"length={length_random}, success={success_random}")
        self.logger.info(f"Deterministic placement (0%): reward={reward_deterministic:.2f}, "
                        f"length={length_deterministic}, success={success_deterministic}")
        
        return results
    
    def run_analysis(self, max_seed_attempts: int = 100, start_seed: int = 0):
        """
        Run the complete counterfactual analysis pipeline.
        
        Args:
            max_seed_attempts: Maximum seeds to try when finding failure case
            start_seed: Starting seed value
            
        Returns:
            Dictionary with complete analysis results
        """
        self.logger.info("Starting coinrun counterfactual analysis")
        
        # Step 1: Find a level seed where weak agent fails with random placement
        failure_seed = self.find_failure_seed(max_seed_attempts, start_seed)
        
        if failure_seed is None:
            self.logger.error("Could not find a failure seed. Analysis cannot continue.")
            return {"error": "No failure seed found"}
        
        # Step 2: Run counterfactual analysis on the failure seed
        results = self.run_counterfactual_analysis(failure_seed)
        
        # Step 3: Save results
        results_file = os.path.join(self.output_dir, f"analysis_results_seed_{failure_seed}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Analysis complete. Results saved to {results_file}")
        return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Coinrun Counterfactual Analysis")
    parser.add_argument(
        "--weak_agent_path",
        default="YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth",
        help="Path to weak agent checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        default="coinrun_analysis_output",
        help="Directory to save results and videos"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        "--max_seed_attempts",
        type=int,
        default=100,
        help="Maximum number of seeds to try when finding failure case"
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="Starting seed value for search"
    )
    parser.add_argument(
        "--check_deps_only",
        action="store_true",
        help="Only check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # If only checking dependencies, exit after the check above
    if args.check_deps_only:
        print("✅ All dependencies are available!")
        sys.exit(0)
    
    # Check if weak agent checkpoint exists
    if not os.path.exists(args.weak_agent_path):
        print(f"Error: Weak agent checkpoint not found at {args.weak_agent_path}")
        print("Please ensure the checkpoint file exists or provide correct path with --weak_agent_path")
        sys.exit(1)
    
    # Create analyzer and run analysis
    try:
        analyzer = CoinrunCounterfactualAnalyzer(
            weak_agent_path=args.weak_agent_path,
            output_dir=args.output_dir,
            device=args.device
        )
        
        results = analyzer.run_analysis(
            max_seed_attempts=args.max_seed_attempts,
            start_seed=args.start_seed
        )
        
        if "error" not in results:
            print("\n=== Analysis Complete ===")
            print(f"Results saved in: {args.output_dir}")
            print(f"Failure seed found: {results['seed']}")
            print(f"Videos saved for both random and deterministic coin placement")
        else:
            print(f"Analysis failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
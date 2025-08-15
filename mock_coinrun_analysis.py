#!/usr/bin/env python3
"""
Mock Coinrun Counterfactual Analysis Demo

This is a simplified demonstration of the coinrun counterfactual analysis logic
that can run without full YRC/procgen dependencies.
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
import random

class MockCoinrunAnalyzer:
    """Mock implementation demonstrating the analysis structure."""
    
    def __init__(self, output_dir: str = "mock_analysis_output"):
        """Initialize the mock analyzer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(output_dir, "coinrun_analysis.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Mock Coinrun Analyzer initialized")
    
    def mock_rollout(self, seed: int, random_percent: int):
        """Mock rollout simulation."""
        # Simulate different outcomes based on random_percent
        # Lower random_percent generally leads to better performance
        random.seed(seed + random_percent)  # Deterministic for testing
        
        if random_percent == 0:  # Deterministic placement
            # Higher chance of success with deterministic placement
            success_prob = 0.8
            reward_base = 10.0
        else:  # Random placement
            # Lower chance of success with random placement  
            success_prob = 0.3
            reward_base = 0.0
        
        success = random.random() < success_prob
        reward = reward_base if success else 0.0
        episode_length = random.randint(50, 200)
        
        return {
            "reward": reward,
            "episode_length": episode_length,
            "success": success,
            "num_frames": episode_length  # Mock frame count
        }
    
    def find_failure_seed(self, max_attempts: int = 100, start_seed: int = 0):
        """Find a seed where the agent fails with random placement."""
        self.logger.info(f"Searching for failure seed (max {max_attempts} attempts)...")
        
        for attempt in range(max_attempts):
            seed = start_seed + attempt
            
            # Test with random placement
            result = self.mock_rollout(seed, random_percent=100)
            success = result["success"]
            
            self.logger.info(f"Seed {seed}: reward={result['reward']:.2f}, "
                           f"length={result['episode_length']}, success={success}")
            
            if not success:
                self.logger.info(f"Found failure seed: {seed}")
                return seed
        
        self.logger.warning(f"No failure seed found in {max_attempts} attempts")
        return None
    
    def run_counterfactual_analysis(self, failure_seed: int):
        """Run mock counterfactual analysis."""
        self.logger.info(f"Running counterfactual analysis for seed {failure_seed}")
        
        results = {
            "seed": failure_seed,
            "timestamp": datetime.now().isoformat(),
            "random_placement": {},
            "deterministic_placement": {}
        }
        
        # Test with random placement
        self.logger.info("Testing with random coin placement (random_percent=100)...")
        random_result = self.mock_rollout(failure_seed, random_percent=100)
        results["random_placement"] = random_result
        
        # Mock video file creation
        random_video_path = os.path.join(self.output_dir, f"seed_{failure_seed}_random_placement.mp4")
        with open(random_video_path, 'w') as f:
            f.write(f"Mock video file - Random placement for seed {failure_seed}\n")
        
        # Test with deterministic placement  
        self.logger.info("Testing with deterministic coin placement (random_percent=0)...")
        deterministic_result = self.mock_rollout(failure_seed, random_percent=0)
        results["deterministic_placement"] = deterministic_result
        
        # Mock video file creation
        det_video_path = os.path.join(self.output_dir, f"seed_{failure_seed}_deterministic_placement.mp4")
        with open(det_video_path, 'w') as f:
            f.write(f"Mock video file - Deterministic placement for seed {failure_seed}\n")
        
        # Log comparison
        self.logger.info("=== Counterfactual Analysis Results ===")
        self.logger.info(f"Seed: {failure_seed}")
        self.logger.info(f"Random placement (100%): reward={random_result['reward']:.2f}, "
                        f"length={random_result['episode_length']}, success={random_result['success']}")
        self.logger.info(f"Deterministic placement (0%): reward={deterministic_result['reward']:.2f}, "
                        f"length={deterministic_result['episode_length']}, success={deterministic_result['success']}")
        
        return results
    
    def run_analysis(self, max_seed_attempts: int = 20, start_seed: int = 0):
        """Run complete mock analysis."""
        self.logger.info("Starting mock coinrun counterfactual analysis")
        
        # Find failure seed
        failure_seed = self.find_failure_seed(max_seed_attempts, start_seed)
        
        if failure_seed is None:
            self.logger.error("Could not find a failure seed in mock analysis")
            return {"error": "No failure seed found"}
        
        # Run counterfactual analysis
        results = self.run_counterfactual_analysis(failure_seed)
        
        # Save results
        results_file = os.path.join(self.output_dir, f"analysis_results_seed_{failure_seed}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Mock analysis complete. Results saved to {results_file}")
        return results

def main():
    """Main function for mock demo."""
    parser = argparse.ArgumentParser(description="Mock Coinrun Counterfactual Analysis Demo")
    parser.add_argument(
        "--output_dir",
        default="mock_analysis_output",
        help="Directory to save mock results"
    )
    parser.add_argument(
        "--max_seed_attempts",
        type=int,
        default=20,
        help="Maximum number of seeds to try"
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="Starting seed value"
    )
    
    args = parser.parse_args()
    
    print("🎮 Mock Coinrun Counterfactual Analysis Demo")
    print("=" * 50)
    print("This is a demonstration of the analysis structure without")
    print("requiring full YRC/procgen dependencies.")
    print("")
    
    try:
        analyzer = MockCoinrunAnalyzer(output_dir=args.output_dir)
        results = analyzer.run_analysis(
            max_seed_attempts=args.max_seed_attempts,
            start_seed=args.start_seed
        )
        
        if "error" not in results:
            print("\n✅ Mock Analysis Complete!")
            print(f"📁 Results saved in: {args.output_dir}")
            print(f"🎯 Failure seed found: {results['seed']}")
            print(f"🎬 Mock videos created for both conditions")
            print("")
            print("📊 Key Finding:")
            random_success = results['random_placement']['success']
            det_success = results['deterministic_placement']['success']
            if not random_success and det_success:
                print("   • Agent failed with random coin placement")
                print("   • Agent succeeded with deterministic coin placement")
                print("   • This suggests coin randomization affected performance!")
            else:
                print(f"   • Random placement success: {random_success}")
                print(f"   • Deterministic placement success: {det_success}")
        else:
            print(f"❌ Mock analysis failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error during mock analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
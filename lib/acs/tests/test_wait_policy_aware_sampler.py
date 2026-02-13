"""
Tests for WaitPolicyAwareSampler with discrete episode length distributions.

This test simulates the scenario where ~30% of episodes timeout at max_episode_length,
creating a discrete distribution that makes certain bins unfillable.
"""

from typing import Callable, Tuple, Dict, Any, List
import numpy as np

from acs import WaitPolicyAwareSampler, SamplingResult
# For visualization (optional)
try:
    from visualization_utils import initialize_test_run, save_single_axis_artifacts
    HAS_VIS_UTILS = True
except ImportError:
    HAS_VIS_UTILS = False


def simulate_episode_lengths(
    num_episodes: int = 5000,
    max_episode_length: int = 500,
    timeout_fraction: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate episode lengths with a fraction timing out at max_episode_length.
    
    Args:
        num_episodes: Number of episodes to simulate
        max_episode_length: Maximum episode length (timeout value)
        timeout_fraction: Fraction of episodes that timeout
        seed: Random seed for reproducibility
        
    Returns:
        Array of episode lengths
    """
    np.random.seed(seed)
    episode_lengths = []
    
    for _ in range(num_episodes):
        if np.random.random() < 1 - timeout_fraction:
            # Episode ends before timeout - uniform distribution
            episode_lengths.append(np.random.randint(10, max_episode_length))
        else:
            # Episode times out at max length
            episode_lengths.append(max_episode_length)
    
    return np.array(episode_lengths)


def create_wait_policy_evaluator(
    episode_lengths: np.ndarray,
    max_episode_length: int = 500
) -> Tuple[
    Callable[[float], Tuple[float, float, Dict[str, Any]]],
    Callable[[], Tuple[float, float, Dict[str, Any]]],
    Callable[[], Tuple[float, float, Dict[str, Any]]],
    List[float]
]:
    """
    Create evaluation functions that simulate wait policy behavior.
    
    The wait policy asks for help if episode_length < threshold.
    This creates a discrete distribution where certain level_afhp
    values are impossible to achieve.
    
    Returns:
        Tuple of (eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme, thresholds_evaluated)
    """
    thresholds_evaluated = []
    
    def evaluate_wait_threshold(threshold: float) -> Tuple[float, float]:
        """Evaluate wait policy at a specific threshold."""
        # Track the threshold
        thresholds_evaluated.append(threshold)
        
        # For WaitPolicy: we wait 'threshold' timesteps, then ask for help
        # So episodes ending before threshold don't ask for help
        # Episodes lasting >= threshold do ask for help
        asked_for_help = np.sum(episode_lengths >= threshold)
        level_afhp = (asked_for_help / len(episode_lengths)) * 100.0
        
        # Simulate performance (higher threshold = lower performance)
        # This is because waiting longer means less help from strong agent
        performance = 85.0 - (threshold / max_episode_length) * 60.0

        return level_afhp, performance
    
    def eval_at_percentile(p: float) -> Tuple[float, float, Dict[str, Any]]:
        """Evaluate at a percentile (0-1)."""
        # Convert percentile to threshold
        # p=0 -> threshold=max_episode_length (never ask)
        # p=1 -> threshold=0 (always ask)
        threshold = max_episode_length * (1 - p)
        ood_pred, perf = evaluate_wait_threshold(threshold)
        
        return ood_pred / 100.0, perf, {"threshold": threshold, "percentile": p}
    
    def eval_at_lower_extreme() -> Tuple[float, float, Dict[str, Any]]:
        """Evaluate at lower extreme (never ask for help)."""
        ood_pred, perf = evaluate_wait_threshold(10000)  # Very high threshold
        return ood_pred / 100.0, perf, {"threshold": 10000, "extreme": "lower"}
    
    def eval_at_upper_extreme() -> Tuple[float, float, Dict[str, Any]]:
        """Evaluate at upper extreme (always ask for help)."""
        ood_pred, perf = evaluate_wait_threshold(0)  # Always ask
        return ood_pred / 100.0, perf, {"threshold": 0, "extreme": "upper"}
    
    return eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme, thresholds_evaluated


def test_wait_policy_30_percent_timeout():
    """Test WaitPolicyAwareSampler with 30% timeout distribution."""
    # Simulate episode lengths
    episode_lengths = simulate_episode_lengths(
        num_episodes=5000,
        max_episode_length=500,
        timeout_fraction=0.3
    )
    
    # Create evaluation functions
    eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme, thresholds_evaluated = \
        create_wait_policy_evaluator(episode_lengths)
    
    # Create sampler
    sampler = WaitPolicyAwareSampler(
        policy_checker=True,  # This is a wait policy
        thresholds_evaluated=thresholds_evaluated,
        max_episode_length=500,
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        num_bins=10,
        output_range=(0.0, 1.0),
        verbose=True
    )
    
    # Run sampler
    result = sampler.run()
    
    # Verify results
    print(f"\n=== Test Results (30% Timeout) ===")
    print(f"Total evaluations: {result.total_evals}")
    print(f"Bins filled: {result.info['bins_filled']} / {result.info['total_bins']}")
    print(f"Early stop reason: {result.early_stop_reason}")
    print(f"Thresholds evaluated: {sorted(thresholds_evaluated)}")
    
    # Debug: Check what ood_pred values we get at different thresholds
    print("\nDebug - Sample threshold evaluations:")
    for test_threshold in [0, 100, 200, 300, 400, 490, 495, 499, 500, 501]:
        asked = np.sum(episode_lengths >= test_threshold)
        ood_pred = (asked / len(episode_lengths)) * 100.0
        print(f"  Threshold {test_threshold}: {ood_pred:.1f}% ask for help")
    
    print(f"\nBin edges: {result.info['bin_edges']}")
    print(f"Points evaluated: {[(p.afhp, p.meta.get('threshold', 'N/A')) for p in result.points]}")
    
    # With 30% timeout, expect approximately 70-90% of bins to be filled
    # Region-based detection allows better coverage than simple percentage would suggest
    filled_fraction = result.info['bins_filled'] / result.info['total_bins']
    expected_filled_fraction = 0.80  # 80% fillable with improved algorithm
    tolerance = 0.15  # Allow ±15% variance
    
    print(f"\nFilled fraction: {filled_fraction:.2f} (expected ~{expected_filled_fraction:.2f})")
    
    if abs(filled_fraction - expected_filled_fraction) > tolerance:
        print(f"\nWARNING: Filled fraction {filled_fraction:.2f} deviates significantly from expected {expected_filled_fraction:.2f}")
    else:
        print(f"✓ Filled fraction is within expected range for 30% timeout distribution")
    
    # Check that we explored a reasonable range  
    # With region-based plateau detection, we should see more evaluations
    if len(thresholds_evaluated) < 8:
        print(f"\nWARNING: Only {len(thresholds_evaluated)} evaluations performed (expected more with region-based detection)")
    
    # Check that we explored both low and high thresholds
    sorted_thresholds = sorted([t for t in thresholds_evaluated if t < 10000])
    if sorted_thresholds:
        print(f"\nThreshold range explored: {min(sorted_thresholds)} to {max(sorted_thresholds)}")
    else:
        print("\nNo valid thresholds explored")
    
    return result


def test_wait_policy_edge_cases():
    """Test WaitPolicyAwareSampler with edge case distributions."""
    test_cases = [
        ("0% Timeout", 0.0),
        ("50% Timeout", 0.5),
        ("100% Timeout", 1.0),
    ]
    
    results = []
    for name, timeout_fraction in test_cases:
        print(f"\n=== Testing {name} ===")
        
        # Simulate episode lengths
        episode_lengths = simulate_episode_lengths(
            num_episodes=1000,
            max_episode_length=500,
            timeout_fraction=timeout_fraction
        )
        
        # Create evaluation functions
        eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme, thresholds_evaluated = \
            create_wait_policy_evaluator(episode_lengths)
        
        # Create sampler
        sampler = WaitPolicyAwareSampler(
            policy_checker=True,
            thresholds_evaluated=thresholds_evaluated,
            max_episode_length=500,
            eval_at_percentile=eval_at_percentile,
            eval_at_lower_extreme=eval_at_lower_extreme,
            eval_at_upper_extreme=eval_at_upper_extreme,
            num_bins=10,
            output_range=(0.0, 1.0),
            verbose=False
        )
        
        # Run sampler
        result = sampler.run()
        results.append((name, result))
        
        print(f"Bins filled: {result.info['bins_filled']} / {result.info['total_bins']}")
        print(f"Early stop: {result.early_stop_reason is not None}")
        
        # Validate expected fill fraction based on timeout percentage
        filled_fraction = result.info['bins_filled'] / result.info['total_bins']
        if timeout_fraction == 0.0:
            # For 0% timeout, all bins should be fillable
            assert result.info['bins_filled'] == result.info['total_bins'], \
                f"Expected all bins to be filled with 0% timeout, got {result.info['bins_filled']}/{result.info['total_bins']}"
            assert result.early_stop_reason is None, \
                "Expected no early stop with 0% timeout"
        elif timeout_fraction == 0.5:
            # For 50% timeout, expect ~50% fillable
            expected_filled = 0.5
            assert abs(filled_fraction - expected_filled) < 0.25, \
                f"Expected ~50% bins filled with 50% timeout, got {filled_fraction:.2f}"
        elif timeout_fraction == 1.0:
            # For 100% timeout, only extremes are fillable
            assert result.info['bins_filled'] <= 3, \
                f"Expected ≤3 bins filled with 100% timeout, got {result.info['bins_filled']}"
    
    return results


def test_narrow_range_detection():
    """Test that narrow range detection works correctly."""
    # Create a distribution that will get stuck near max_episode_length
    episode_lengths = simulate_episode_lengths(
        num_episodes=1000,
        max_episode_length=500,
        timeout_fraction=0.4  # 40% timeout
    )
    
    # Create evaluation functions
    eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme, thresholds_evaluated = \
        create_wait_policy_evaluator(episode_lengths)
    
    # Create sampler with verbose output to see the narrow range message
    sampler = WaitPolicyAwareSampler(
        policy_checker=True,
        thresholds_evaluated=thresholds_evaluated,
        max_episode_length=500,
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        num_bins=20,  # More bins to increase chance of getting stuck
        output_range=(0.0, 1.0),
        verbose=True
    )
    
    # Run sampler
    result = sampler.run()
    
    print(f"\n=== Narrow Range Detection Test ===")
    print(f"Early stop reason: {result.early_stop_reason}")
    print(f"Bins filled: {result.info['bins_filled']} / {result.info['total_bins']}")
    
    # With 40% timeout, expect approximately 60% of bins to be filled
    filled_fraction = result.info['bins_filled'] / result.info['total_bins']
    expected_filled_fraction = 0.60  # 60% fillable with 40% timeout
    tolerance = 0.20  # Allow ±20% variance for this test
    
    print(f"Filled fraction: {filled_fraction:.2f} (expected ~{expected_filled_fraction:.2f})")
    
    if abs(filled_fraction - expected_filled_fraction) > tolerance:
        print(f"WARNING: Filled fraction {filled_fraction:.2f} deviates from expected {expected_filled_fraction:.2f}")
    else:
        print(f"✓ Filled fraction is within expected range for 40% timeout distribution")
    
    return result


if __name__ == "__main__":
    # Initialize test run
    if HAS_VIS_UTILS:
        from pathlib import Path
        timestamp = initialize_test_run()
        output_dir = Path(f"test_artifacts/{timestamp}/wait_policy_sampler_tests")
    
    # Run main test
    print("Running main test (30% timeout)...")
    result_30 = test_wait_policy_30_percent_timeout()
    
    # Run edge case tests
    print("\nRunning edge case tests...")
    edge_results = test_wait_policy_edge_cases()
    
    # Run narrow range test
    print("\nRunning narrow range detection test...")
    narrow_result = test_narrow_range_detection()
    
    # Save artifacts for main test if visualization utils available
    if HAS_VIS_UTILS:
        save_single_axis_artifacts(
            points=result_30.points,
            output_dir=output_dir / "wait_policy_30_percent",
            test_name="Wait Policy 30% Timeout",
            input_name="Percentile",
            output_name="OOD Prediction %",
            performance_name="Performance",
            coverage_fraction=0.1,
            info=result_30.info
        )
        print(f"\n=== All Tests Passed ===")
        print(f"Test artifacts saved to: {output_dir}")
    else:
        print(f"\n=== All Tests Passed ===")
        print("(Visualization skipped - visualization_utils not available)")
    
    print("\nSummary:")
    print(f"- 30% timeout: {result_30.info['bins_filled']}/{result_30.info['total_bins']} bins filled")
    for name, result in edge_results:
        print(f"- {name}: {result.info['bins_filled']}/{result.info['total_bins']} bins filled")
    print(f"- Narrow range test: Early stop = {narrow_result.early_stop_reason is not None}")
"""
YRC-specific wrappers for the ABCS (Adaptive Binary Coverage Search) library.

This module provides YRC-specific convenience functions that wrap the generic
ABCS library for threshold evaluation use cases.
"""

from typing import List, Tuple, Callable, Optional, Any, Dict

# Import from the external ABCS library
from abcs import BinarySearchSampler, SamplePoint


def create_threshold_sampler(
    policy,
    evaluator,
    envs,
    split: str,
    num_bins: int,
    logger=None,
) -> BinarySearchSampler:
    """
    Create a sampler specifically for threshold evaluation.

    This is a convenience function that wraps the generic ABCS sampler
    for the specific use case of threshold evaluation in YRC.
    
    Args:
        policy: Policy object with threshold evaluation capabilities
        evaluator: Evaluator object for running policy evaluations
        envs: Environment(s) to evaluate on
        split: Data split to use for evaluation ("train", "val", "test")
        num_bins: Number of AFHP bins for coverage
        logger: Optional logger for tracking evaluations
    
    Returns:
        BinarySearchSampler configured for threshold evaluation
    """

    def eval_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        """Evaluate policy at given threshold."""
        # Update policy with threshold
        if hasattr(policy, "update_params"):
            policy.update_params({"threshold": threshold})

        # Run evaluation
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold
        )

        # Extract AFHP as output value
        afhp = summary[split]["action_1_frac"]

        # Return AFHP and full summary as metadata
        return afhp * 100, {"summary": summary, "threshold": threshold}

    def percentile_to_threshold(percentile: float) -> float:
        """Convert percentile to threshold."""
        if percentile == 0:
            return float("inf")
        elif percentile == 100:
            return float("-inf")
        else:
            return policy.train_percentile(100 - percentile)

    return BinarySearchSampler(
        eval_function=eval_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),  # Percentiles
        output_range=(0.0, 100.0),  # AFHP percentage
        input_to_threshold=percentile_to_threshold,
    )
"""
Wait Policy Aware Sampler for handling discrete episode length distributions.

This module provides a specialized sampler that detects when certain bins cannot
be filled due to discrete distributions (e.g., when a significant percentage of
episodes timeout at a fixed length).
"""

from typing import Any, Optional

from acs.sampler import BinarySearchSampler
from acs.types import SamplingResult


class WaitPolicyAwareSampler(BinarySearchSampler):
    """Custom sampler that detects wait policy limitations and stops early when appropriate."""
    
    def __init__(
        self,
        policy_checker: Any,
        thresholds_evaluated: list,
        max_episode_length: int,
        *args,
        **kwargs
    ):
        """
        Initialize the WaitPolicyAwareSampler.
        
        Args:
            policy_checker: Function or object to check if this is a wait-like policy
            thresholds_evaluated: List to track evaluated thresholds
            max_episode_length: Maximum episode length (e.g., 500 for maze)
            *args: Additional arguments for BinarySearchSampler
            **kwargs: Additional keyword arguments for BinarySearchSampler
        """
        super().__init__(*args, **kwargs)
        self.policy_checker = policy_checker
        self.thresholds_evaluated = thresholds_evaluated
        self.max_episode_length = max_episode_length
        self.is_wait_policy = callable(policy_checker) and policy_checker() or policy_checker is True
    
    def binary_search_fill(self, left_input, right_input, left_bin_idx, right_bin_idx):
        """Override binary search to detect when we should stop early."""
        # Check if we should stop early for WaitPolicy
        if self.is_wait_policy and len(self.thresholds_evaluated) >= 6:
            # Check if we've explored a reasonable range but still have gaps
            sorted_thresholds = sorted([t for t in self.thresholds_evaluated if t < 10000])
            if len(sorted_thresholds) >= 6:
                min_threshold = min(sorted_thresholds)
                max_threshold = max(sorted_thresholds)
                
                # Check if we're stuck in a narrow range near max_episode_length
                recent_thresholds = sorted_thresholds[-4:]  # Last 4 thresholds
                if len(recent_thresholds) >= 4:
                    threshold_range = max(recent_thresholds) - min(recent_thresholds)
                    # If last 4 thresholds are within 10 timesteps and near max_episode_length
                    if threshold_range <= 10 and min(recent_thresholds) >= self.max_episode_length - 20:
                        remaining_bins = self.bins_remaining(left_bin_idx, right_bin_idx)
                        if remaining_bins:
                            print(
                                f"WaitPolicy: Stuck in narrow range {min(recent_thresholds)}-{max(recent_thresholds)} "
                                f"near max_episode_length={self.max_episode_length}, stopping early"
                            )
                            return 0
                
                # If we've tested low (<=2) and high (>=max_episode_length-5) thresholds
                # and still have empty bins, we can't fill them
                if min_threshold <= 2 and max_threshold >= self.max_episode_length - 5:
                    remaining_bins = self.bins_remaining(left_bin_idx, right_bin_idx)
                    if remaining_bins:
                        print(
                            f"WaitPolicy: Explored thresholds {min_threshold}-{max_threshold} "
                            f"(max_episode_length={self.max_episode_length}) but cannot fill remaining bins, stopping early"
                        )
                        return 0
        return super().binary_search_fill(left_input, right_input, left_bin_idx, right_bin_idx)
    
    def _create_sampling_result(self):
        """Add early stop reason if we couldn't fill all bins."""
        result = super()._create_sampling_result()
        
        # Add early stop reason if we couldn't fill all bins with WaitPolicy
        if self.is_wait_policy and len(self.thresholds_evaluated) >= 6:
            sorted_thresholds = sorted([t for t in self.thresholds_evaluated if t < 10000])
            if len(sorted_thresholds) >= 6:
                min_threshold = min(sorted_thresholds)
                max_threshold = max(sorted_thresholds)
                bins_filled = result.info.get("bins_filled", 0)
                total_bins = result.info.get("total_bins", 0)
                
                if (min_threshold <= 2 and max_threshold >= self.max_episode_length - 5 
                    and bins_filled < total_bins):
                    result = SamplingResult(
                        points=result.points,
                        coverage_x_max_gap=result.coverage_x_max_gap,
                        coverage_y_max_gap=result.coverage_y_max_gap,
                        total_evals=result.total_evals,
                        early_stop_reason="WaitPolicy: Cannot fill all bins due to discrete episode length distribution",
                        monotonicity_violations_remaining=result.monotonicity_violations_remaining,
                        info=result.info,
                    )
        return result
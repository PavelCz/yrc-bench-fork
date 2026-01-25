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
        self.detected_unfillable_region = False
    
    def binary_search_fill(self, left_input, right_input, left_bin_idx, right_bin_idx):
        """Override binary search to detect when we should stop early."""
        # If the search range is very narrow, don't continue
        if abs(right_input - left_input) < 1e-6:
            return 0
            
        # Check if we're getting multiple samples with the same output value
        if self.is_wait_policy and len(self.all_samples) >= 4:
            # Look at recent samples to detect if we're stuck
            recent_samples = self.all_samples[-3:]
            recent_outputs = [s.afhp for s in recent_samples]
            
            # If last 3 samples have very similar output values (within 2% of range)
            output_range_tolerance = 0.02  # 2% of [0,1] range
            if max(recent_outputs) - min(recent_outputs) < output_range_tolerance:
                # Check corresponding thresholds to see if we're exploring a narrow range
                recent_thresholds = []
                for s in recent_samples:
                    if s.meta and 'threshold' in s.meta:
                        threshold = s.meta['threshold']
                        if threshold < 10000:  # Exclude infinity placeholders
                            recent_thresholds.append(threshold)
                
                if len(recent_thresholds) >= 3:
                    # Check if we're in a narrow threshold range relative to max_episode_length
                    threshold_range = max(recent_thresholds) - min(recent_thresholds)
                    # If exploring less than 5% of the episode length range
                    if threshold_range < 0.05 * self.max_episode_length:
                        # We're stuck exploring a narrow threshold range with no output change
                        self.detected_unfillable_region = True
                        remaining = self.bins_remaining(left_bin_idx, right_bin_idx)
                        if remaining and self.verbose:
                            avg_output = sum(recent_outputs) / len(recent_outputs)
                            print(f"WaitPolicy: Detected plateau at output ~{avg_output:.2f} "
                                  f"(thresholds {min(recent_thresholds):.1f}-{max(recent_thresholds):.1f}), "
                                  f"skipping narrow search")
                        return 0
        
        return super().binary_search_fill(left_input, right_input, left_bin_idx, right_bin_idx)
    
    def _create_sampling_result(self):
        """Add early stop reason if we couldn't fill all bins."""
        result = super()._create_sampling_result()
        
        # Add info about unfillable regions if detected
        if self.is_wait_policy and self.detected_unfillable_region:
            bins_filled = result.info.get("bins_filled", 0)
            total_bins = result.info.get("total_bins", 0)
            
            if bins_filled < total_bins:
                # Update the info to mention partial coverage due to discrete distribution
                result.info["partial_coverage_reason"] = "WaitPolicy: Some bins unfillable due to discrete episode length distribution"
                
                # Only set early_stop_reason if we think this significantly impacted coverage
                sorted_thresholds = sorted([t for t in self.thresholds_evaluated if t < 10000])
                if len(sorted_thresholds) >= 6:
                    min_threshold = min(sorted_thresholds)
                    max_threshold = max(sorted_thresholds)
                    
                    # If we explored most of the threshold range but still have significant gaps
                    threshold_coverage = (max_threshold - min_threshold) / self.max_episode_length
                    bin_coverage = bins_filled / total_bins
                    
                    # If we covered >80% of threshold range but <80% of bins
                    if threshold_coverage > 0.8 and bin_coverage < 0.8:
                        result = SamplingResult(
                            points=result.points,
                            coverage_x_max_gap=result.coverage_x_max_gap,
                            coverage_y_max_gap=result.coverage_y_max_gap,
                            total_evals=result.total_evals,
                            early_stop_reason=f"WaitPolicy: Only {bins_filled}/{total_bins} bins filled despite exploring {threshold_coverage:.0%} of threshold range",
                            monotonicity_violations_remaining=result.monotonicity_violations_remaining,
                            info=result.info,
                        )
        return result
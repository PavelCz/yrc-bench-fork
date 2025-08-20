"""
YRC-specific wrappers for the ACS (Adaptive Coverage Sampling) library.

This module provides YRC-specific convenience functions that wrap the generic
ACS library for the specific use case of threshold evaluation in YRC.
"""

from typing import Tuple, Any, Dict

# Import the joint-coverage sampler from the external ACS library
from acs import BinarySearchSampler
from YRC.policies.ood import OODPolicy
from YRC.policies.lightning_ae import LightningAEPolicy
from YRC.policies.base import RandomPolicy
from YRC.core import Evaluator


def create_threshold_sampler(
    policy,
    evaluator: Evaluator,
    envs,
    split: str,
    *,
    coverage_fraction: float = 0.10,
    max_total_evals: int = 200,
    logger=None,
):
    """
    Create the joint-coverage sampler for threshold evaluation.

    This wrapper adapts YRC evaluation to the ACS JointCoverageSampler API by
    providing evaluation callables for percentiles and extremes, and also
    records the latest summaries and thresholds used per percentile.

    Args:
        policy: Policy object with threshold evaluation capabilities
        evaluator: Evaluator object for running policy evaluations
        envs: Environment(s) to evaluate on
        split: Data split to use for evaluation ("train", "val", "test")
        coverage_fraction: Maximum allowed normalized neighbor gap on both axes
        max_total_evals: Global evaluation budget (includes re-runs)
        logger: Optional logger for tracking evaluations

    Returns:
        JointCoverageSampler ready to run
    """

    def percentile_to_threshold(p: float) -> float:
        # p in [0,1]
        if p <= 0.0:
            return float("inf")
        if p >= 1.0:
            return float("-inf")
        return policy.train_percentile(100.0 - (p * 100.0))

    def _eval_with_threshold(threshold: float) -> Tuple[float, float, Dict[str, Any]]:
        update_policy_params(policy, threshold)
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold
        )
        afhp = summary[split]["action_1_frac"] * 100.0
        performance = float(summary[split]["env_reward_mean"])  # Y-axis
        return afhp, performance, {"summary": summary, "threshold": threshold}

    def eval_at_percentile(p: float) -> Tuple[float, float, Dict[str, Any]]:
        thr = percentile_to_threshold(p)
        afhp, performance, meta = _eval_with_threshold(thr)
        return afhp, performance, meta

    def eval_at_lower_extreme() -> Tuple[float, float, Dict[str, Any]]:
        thr = float("inf")
        afhp, performance, meta = _eval_with_threshold(thr)
        return afhp, performance, meta

    def eval_at_upper_extreme() -> Tuple[float, float, Dict[str, Any]]:
        thr = float("-inf")
        afhp, performance, meta = _eval_with_threshold(thr)
        return afhp, performance, meta
    
    # Convert coverate fraction to num_bins
    num_bins = int(1.0 / coverage_fraction)

    return BinarySearchSampler(
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        num_bins=num_bins,
        # max_total_evals=max_total_evals,
    )


def update_policy_params(policy, threshold):
    if isinstance(policy, LightningAEPolicy) or isinstance(policy, OODPolicy):
        policy.update_params({"threshold": threshold})
    elif isinstance(policy, RandomPolicy):
        if threshold == float("inf"):
            # An infinite threshold means that the policy will never ask for help.
            # We need to set the probability to 0.
            threshold = 0.0
        elif threshold == float("-inf"):
            # A negative infinite threshold means that the policy will always ask for help.
            # We need to set the probability to 1.
            threshold = 1.0
        policy.update_params(threshold)

    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )

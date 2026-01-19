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
from YRC.policies.base import TimestepRandomPolicy, LevelBasedRandomPolicy
from YRC.policies.threshold import ThresholdPolicy
from YRC.policies.heuristic import ExponentialHeuristicPolicy
from YRC.core import Evaluator
import numpy as np


def create_ood_percentage_threshold_sampler(
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
        # Don't close environments between evaluations - the sampler runs multiple evals
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold, close_envs=False
        )
        level_ood_preds = summary[split]["level_ood_pred"]
        target_metric = float(np.mean(level_ood_preds))
        performance = float(summary[split]["env_return_mean"])  # Y-axis
        return target_metric, performance, {"summary": summary, "threshold": threshold}

    def eval_at_percentile(p: float) -> Tuple[float, float, Dict[str, Any]]:
        thr = percentile_to_threshold(p)
        target_metric, performance, meta = _eval_with_threshold(thr)
        return target_metric, performance, meta

    def eval_at_lower_extreme() -> Tuple[float, float, Dict[str, Any]]:
        thr = float("inf")
        target_metric, performance, meta = _eval_with_threshold(thr)
        return target_metric, performance, meta

    def eval_at_upper_extreme() -> Tuple[float, float, Dict[str, Any]]:
        thr = float("-inf")
        target_metric, performance, meta = _eval_with_threshold(thr)
        return target_metric, performance, meta

    # Convert coverate fraction to num_bins
    num_bins = int(1.0 / coverage_fraction)

    return BinarySearchSampler(
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        num_bins=num_bins,
        # AFHP uses [0, 100] interval, here we use [0, 1]
        output_range=(0.0, 1.0),
        # max_total_evals=max_total_evals,
    )


def create_afhp_threshold_sampler(
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
        # Don't close environments between evaluations - the sampler runs multiple evals
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold, close_envs=False
        )
        afhp = summary[split]["action_1_frac"] * 100.0
        performance = float(summary[split]["env_return_mean"])  # Y-axis
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
    if (
        isinstance(policy, LightningAEPolicy)
        or isinstance(policy, OODPolicy)
        or isinstance(policy, ThresholdPolicy)
    ):
        params = policy.params.copy()
        params["threshold"] = threshold

        policy.update_params(params)
    elif (
        isinstance(policy, TimestepRandomPolicy)
        or isinstance(policy, LevelBasedRandomPolicy)
        or isinstance(policy, ExponentialHeuristicPolicy)
    ):
        if threshold == float("inf"):
            # An infinite threshold means that the policy will never ask for help.
            # We need to set the probability to 0.
            threshold = 0.0
        elif threshold == float("-inf"):
            # A negative infinite threshold means that the policy will always ask for
            # help.
            # We need to set the probability to 1.
            threshold = 1.0
        policy.update_params(threshold)

    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )

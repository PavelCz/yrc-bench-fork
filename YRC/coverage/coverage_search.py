"""
YRC-specific wrappers for the ACS (Adaptive Coverage Sampling) library.

This module provides YRC-specific convenience functions that wrap the generic
ACS library for the specific use case of threshold evaluation in YRC.
"""

from typing import Tuple, Any, Dict, Optional

# Import the joint-coverage sampler from the external ACS library
from acs import BinarySearchSampler
from acs.wait_policy_sampler import WaitPolicyAwareSampler
from YRC.policies.ood import OODPolicy
from YRC.policies.lightning_ae import LightningAEPolicy
from YRC.policies.base import (
    LevelBasedRandomPolicy,
    OracleLevelBasedRandomPolicy,
    TimestepRandomPolicy,
)
from YRC.policies.threshold import ThresholdPolicy
from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy
from YRC.core import Evaluator
import numpy as np
import logging

try:
    import wandb
except ImportError:
    wandb = None


class EvalStepTracker:
    """Tracks evaluation steps and logs metrics to wandb."""

    def __init__(self, wandb_run: Optional[Any] = None):
        self.step = 0
        self.wandb_run = wandb_run

    def log_eval(
        self,
        threshold: float,
        step_afhp: float,
        level_afhp: float,
        performance: float,
    ):
        """Log evaluation metrics to console and wandb."""
        self.step += 1

        # Print to console
        message = (
            f"[Eval {self.step:3d}] threshold={threshold:10.4f}, "
            f"step_afhp={step_afhp:6.2f}%, level_afhp={level_afhp:6.2f}%, "
            f"performance={performance:.4f}"
        )
        print(message)
        logging.info(message)

        # Log to wandb
        if self.wandb_run is not None and wandb is not None:
            wandb.log(
                {
                    "eval/step": self.step,
                    "eval/threshold": threshold
                    if not np.isinf(threshold)
                    else (1e10 if threshold > 0 else -1e10),
                    "eval/step_afhp": step_afhp,
                    "eval/level_afhp": level_afhp,
                    "eval/performance": performance,
                },
                step=self.step,
            )


def create_level_afhp_threshold_sampler(
    policy,
    evaluator: Evaluator,
    envs_factory,
    split: str,
    *,
    coverage_fraction: float = 0.10,
    max_total_evals: int = 200,
    logger=None,
    wandb_run=None,
):
    """
    Create the joint-coverage sampler for threshold evaluation.

    This wrapper adapts YRC evaluation to the ACS JointCoverageSampler API by
    providing evaluation callables for percentiles and extremes, and also
    records the latest summaries and thresholds used per percentile.

    Args:
        policy: Policy object with threshold evaluation capabilities
        evaluator: Evaluator object for running policy evaluations
        envs_factory: Callable that returns fresh environments, ensuring each
            evaluation sees the same seeds in the same order
        split: Data split to use for evaluation ("train", "val", "test")
        coverage_fraction: Maximum allowed normalized neighbor gap on both axes
        max_total_evals: Global evaluation budget (includes re-runs)
        logger: Optional logger for tracking evaluations
        wandb_run: Optional wandb run for logging metrics

    Returns:
        JointCoverageSampler ready to run
    """
    tracker = EvalStepTracker(wandb_run=wandb_run)

    # Track threshold values seen for WaitPolicy
    thresholds_evaluated = []

    # Get max episode length for WaitPolicy
    max_episode_length = None
    if isinstance(policy, WaitPolicy):
        max_episode_length = getattr(policy, "max_episode_length", 500)

    def percentile_to_threshold(p: float) -> float:
        # p in [0,1]
        if p <= 0.0:
            return float("inf")
        if p >= 1.0:
            return float("-inf")
        percentile = 100.0 - (p * 100.0)
        return policy.train_percentile_level(percentile)

    def _eval_with_threshold(threshold: float) -> Tuple[float, float, Dict[str, Any]]:
        update_policy_params(policy, threshold)

        # Track thresholds for WaitPolicy
        if isinstance(policy, WaitPolicy) and max_episode_length is not None:
            # Convert inf thresholds to actual values for tracking
            actual_threshold = threshold
            if threshold == float("inf"):
                actual_threshold = 10000
            elif threshold == float("-inf"):
                actual_threshold = 0
            thresholds_evaluated.append(actual_threshold)

        # Create fresh environments for each evaluation to ensure reproducibility
        # Each evaluation sees the same seeds in the same order
        envs = envs_factory()
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold, close_envs=True
        )
        level_ood_preds = summary[split]["level_ood_pred"]
        level_afhp = float(np.mean(level_ood_preds)) * 100.0
        step_afhp = summary[split]["action_1_frac"] * 100.0
        performance = float(summary[split]["env_return_mean"])  # Y-axis

        # Log to console and wandb
        tracker.log_eval(
            threshold=threshold,
            step_afhp=step_afhp,
            level_afhp=level_afhp,
            performance=performance,
        )

        # Return level_afhp in [0, 1] for the sampler
        target_metric = level_afhp / 100.0
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

    # Use WaitPolicyAwareSampler if we have a WaitPolicy, otherwise use regular BinarySearchSampler
    if isinstance(policy, WaitPolicy):
        return WaitPolicyAwareSampler(
            policy_checker=lambda: isinstance(policy, WaitPolicy),
            thresholds_evaluated=thresholds_evaluated,
            max_episode_length=max_episode_length,
            eval_at_percentile=eval_at_percentile,
            eval_at_lower_extreme=eval_at_lower_extreme,
            eval_at_upper_extreme=eval_at_upper_extreme,
            num_bins=num_bins,
            # AFHP uses [0, 100] interval, here we use [0, 1]
            output_range=(0.0, 1.0),
        )
    else:
        return BinarySearchSampler(
            eval_at_percentile=eval_at_percentile,
            eval_at_lower_extreme=eval_at_lower_extreme,
            eval_at_upper_extreme=eval_at_upper_extreme,
            num_bins=num_bins,
            # AFHP uses [0, 100] interval, here we use [0, 1]
            output_range=(0.0, 1.0),
        )


def create_step_afhp_threshold_sampler(
    policy,
    evaluator: Evaluator,
    envs_factory,
    split: str,
    *,
    coverage_fraction: float = 0.10,
    max_total_evals: int = 200,
    logger=None,
    wandb_run=None,
):
    """
    Create the joint-coverage sampler for threshold evaluation.

    This wrapper adapts YRC evaluation to the ACS JointCoverageSampler API by
    providing evaluation callables for percentiles and extremes, and also
    records the latest summaries and thresholds used per percentile.

    Args:
        policy: Policy object with threshold evaluation capabilities
        evaluator: Evaluator object for running policy evaluations
        envs_factory: Callable that returns fresh environments, ensuring each
            evaluation sees the same seeds in the same order
        split: Data split to use for evaluation ("train", "val", "test")
        coverage_fraction: Maximum allowed normalized neighbor gap on both axes
        max_total_evals: Global evaluation budget (includes re-runs)
        logger: Optional logger for tracking evaluations
        wandb_run: Optional wandb run for logging metrics

    Returns:
        JointCoverageSampler ready to run
    """
    tracker = EvalStepTracker(wandb_run=wandb_run)

    # Track threshold values seen for WaitPolicy
    thresholds_evaluated = []

    # Get max episode length for WaitPolicy
    max_episode_length = None
    if isinstance(policy, WaitPolicy):
        max_episode_length = getattr(policy, "max_episode_length", 500)

    def percentile_to_threshold(p: float) -> float:
        # p in [0,1]
        if p <= 0.0:
            return float("inf")
        if p >= 1.0:
            return float("-inf")
        percentile = 100.0 - (p * 100.0)
        return policy.train_percentile_step(percentile)

    def _eval_with_threshold(threshold: float) -> Tuple[float, float, Dict[str, Any]]:
        update_policy_params(policy, threshold)

        # Track thresholds for WaitPolicy
        if isinstance(policy, WaitPolicy) and max_episode_length is not None:
            # Convert inf thresholds to actual values for tracking
            actual_threshold = threshold
            if threshold == float("inf"):
                actual_threshold = 10000
            elif threshold == float("-inf"):
                actual_threshold = 0
            thresholds_evaluated.append(actual_threshold)

        # Create fresh environments for each evaluation to ensure reproducibility
        # Each evaluation sees the same seeds in the same order
        envs = envs_factory()
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold, close_envs=True
        )
        level_ood_preds = summary[split]["level_ood_pred"]
        level_afhp = float(np.mean(level_ood_preds)) * 100.0
        step_afhp = summary[split]["action_1_frac"] * 100.0
        performance = float(summary[split]["env_return_mean"])  # Y-axis

        # Log to console and wandb
        tracker.log_eval(
            threshold=threshold,
            step_afhp=step_afhp,
            level_afhp=level_afhp,
            performance=performance,
        )

        return step_afhp, performance, {"summary": summary, "threshold": threshold}

    def eval_at_percentile(p: float) -> Tuple[float, float, Dict[str, Any]]:
        thr = percentile_to_threshold(p)
        step_afhp, performance, meta = _eval_with_threshold(thr)
        return step_afhp, performance, meta

    def eval_at_lower_extreme() -> Tuple[float, float, Dict[str, Any]]:
        thr = float("inf")
        step_afhp, performance, meta = _eval_with_threshold(thr)
        return step_afhp, performance, meta

    def eval_at_upper_extreme() -> Tuple[float, float, Dict[str, Any]]:
        thr = float("-inf")
        step_afhp, performance, meta = _eval_with_threshold(thr)
        return step_afhp, performance, meta

    # Convert coverate fraction to num_bins
    num_bins = int(1.0 / coverage_fraction)

    # Use WaitPolicyAwareSampler if we have a WaitPolicy, otherwise use regular BinarySearchSampler
    if isinstance(policy, WaitPolicy):
        return WaitPolicyAwareSampler(
            policy_checker=lambda: isinstance(policy, WaitPolicy),
            thresholds_evaluated=thresholds_evaluated,
            max_episode_length=max_episode_length,
            eval_at_percentile=eval_at_percentile,
            eval_at_lower_extreme=eval_at_lower_extreme,
            eval_at_upper_extreme=eval_at_upper_extreme,
            num_bins=num_bins,
        )
    else:
        return BinarySearchSampler(
            eval_at_percentile=eval_at_percentile,
            eval_at_lower_extreme=eval_at_lower_extreme,
            eval_at_upper_extreme=eval_at_upper_extreme,
            num_bins=num_bins,
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
        or isinstance(policy, OracleLevelBasedRandomPolicy)
        or isinstance(policy, ExponentialHeuristicPolicy)
    ):
        if threshold == float("inf"):
            # An infinite threshold means that the policy will never ask for help.
            # We need to set the control parameter to 0.
            threshold = 0.0
        elif threshold == float("-inf"):
            # A negative infinite threshold means that the policy will always ask for
            # help.
            # OracleLevelBasedRandomPolicy uses 2.0 as the always-ask setting.
            threshold = 2.0 if isinstance(policy, OracleLevelBasedRandomPolicy) else 1.0
        policy.update_params(threshold)

    elif isinstance(policy, WaitPolicy):
        if threshold == float("inf"):
            # Never ask for help - set very high timestep threshold
            threshold = 10000
        elif threshold == float("-inf"):
            # Always ask for help - set threshold to 0 (ask from start)
            threshold = 0
        policy.update_params(threshold=threshold)

    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )

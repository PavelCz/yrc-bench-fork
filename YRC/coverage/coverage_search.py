"""
YRC-specific wrappers for the ACS (Adaptive Coverage Sampling) library.

This module provides YRC-specific convenience functions that wrap the generic
ACS library for the specific use case of threshold evaluation in YRC.
"""

from typing import Tuple, Any, Dict, Callable

# Import the joint-coverage sampler from the external ACS library
from abcs import JointCoverageSampler


def create_threshold_sampler(
    policy,
    evaluator,
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
        An adapter exposing `.run() -> SamplingResult` and `.get_metadata()`.
    """

    # Stores the most recent summary/threshold per percentile (keyed by float p in [0,1])
    last_summaries: Dict[float, Any] = {}
    last_thresholds: Dict[float, float] = {}

    def percentile_to_threshold(p: float) -> float:
        # p in [0,1]
        if p <= 0.0:
            return float("inf")
        if p >= 1.0:
            return float("-inf")
        return policy.train_percentile(100.0 - (p * 100.0))

    def _eval_with_threshold(threshold: float) -> Tuple[float, Dict[str, Any]]:
        if hasattr(policy, "update_params"):
            policy.update_params({"threshold": threshold})
        summary = evaluator.eval(
            policy, envs, [split], logger=logger, threshold=threshold
        )
        afhp = summary[split]["action_1_frac"] * 100.0
        performance = float(summary[split]["env_reward_mean"])  # Y-axis
        return afhp, {"summary": summary, "threshold": threshold, "performance": performance}

    def eval_at_percentile(p: float) -> Tuple[float, float]:
        thr = percentile_to_threshold(p)
        afhp, meta = _eval_with_threshold(thr)
        last_summaries[p] = meta["summary"]
        last_thresholds[p] = thr
        return afhp, meta["performance"]

    def eval_at_lower_extreme() -> Tuple[float, float]:
        thr = float("inf")
        afhp, meta = _eval_with_threshold(thr)
        last_summaries[0.0] = meta["summary"]
        last_thresholds[0.0] = thr
        return afhp, meta["performance"]

    def eval_at_upper_extreme() -> Tuple[float, float]:
        thr = float("-inf")
        afhp, meta = _eval_with_threshold(thr)
        last_summaries[1.0] = meta["summary"]
        last_thresholds[1.0] = thr
        return afhp, meta["performance"]

    sampler = JointCoverageSampler(
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        coverage_fraction=coverage_fraction,
        max_total_evals=max_total_evals,
    )

    class ThresholdSamplerAdapter:
        def __init__(self, inner, summaries, thresholds):
            self._inner = inner
            self._summaries = summaries
            self._thresholds = thresholds

        def run(self):
            return self._inner.run()

        def get_metadata(self) -> Tuple[Dict[float, Any], Dict[float, float]]:
            return self._summaries, self._thresholds

    return ThresholdSamplerAdapter(sampler, last_summaries, last_thresholds)
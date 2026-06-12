"""
YRC-specific wrappers for the ACS (Adaptive Coverage Sampling) library.

This module provides YRC-specific convenience functions that wrap the generic
ACS library for the specific use case of threshold evaluation in YRC.
"""

from typing import Tuple, Any, Dict, Optional, Callable, List

# Import the joint-coverage sampler from the external ACS library
from acs import BinarySearchSampler
from acs.types import CurvePoint, SamplingResult
from acs.wait_policy_sampler import WaitPolicyAwareSampler
from YRC.policies.ood import OODPolicy
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


IMAGE_SVDD_DEGENERATE_STRATEGY = "expand_above_id"
DEFAULT_IMAGE_SVDD_EXPANSION_MAX_EVALS = 12
DEFAULT_IMAGE_SVDD_EXPANSION_INITIAL_DELTA_FRACTION = 1e-4
DEFAULT_IMAGE_SVDD_PROBE_PERCENTILES = (0.25, 0.5, 0.75, 0.9)
DEFAULT_IMAGE_SVDD_HIGH_AFHP_THRESHOLD = 0.99
DEFAULT_SCORE_TOLERANCE_ABS = 1e-8
DEFAULT_SCORE_TOLERANCE_REL = 1e-6


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
            f"[Eval {self.step:3d}] threshold={threshold:14.8g}, "
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


def image_svdd_calibration_diagnostics(
    policy,
    *,
    score_tolerance_abs: float = DEFAULT_SCORE_TOLERANCE_ABS,
    score_tolerance_rel: float = DEFAULT_SCORE_TOLERANCE_REL,
) -> Dict[str, Any]:
    """Return calibration-collapse diagnostics for image DeepSVDD policies."""
    diagnostics = {
        "is_image_svdd": False,
        "is_degenerate": False,
        "num_scores": 0,
        "num_finite_scores": 0,
        "unique_count": 0,
        "min_score": None,
        "max_score": None,
        "score_range": None,
        "tolerance": None,
    }

    if not isinstance(policy, OODPolicy):
        return diagnostics
    if getattr(policy, "clf_name", None) != "DeepSVDD":
        return diagnostics
    if getattr(policy, "feature_type", None) != "obs":
        return diagnostics

    diagnostics["is_image_svdd"] = True
    scores = getattr(policy, "_train_episode_max_scores", None)
    if scores is None:
        return diagnostics

    scores = np.asarray(scores, dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    diagnostics["num_scores"] = int(scores.size)
    diagnostics["num_finite_scores"] = int(finite_scores.size)
    if finite_scores.size == 0:
        return diagnostics

    min_score = float(np.min(finite_scores))
    max_score = float(np.max(finite_scores))
    score_range = float(max_score - min_score)
    tolerance = float(max(score_tolerance_abs, abs(max_score) * score_tolerance_rel))
    sorted_scores = np.sort(finite_scores)
    unique_count = int(1 + np.count_nonzero(np.diff(sorted_scores) > tolerance))

    diagnostics.update(
        {
            "unique_count": unique_count,
            "min_score": min_score,
            "max_score": max_score,
            "score_range": score_range,
            "tolerance": tolerance,
            "is_degenerate": unique_count < 2 or score_range <= tolerance,
        }
    )
    return diagnostics


def _unique_count(values: np.ndarray, tolerance: float) -> int:
    if values.size == 0:
        return 0
    sorted_values = np.sort(values)
    return int(1 + np.count_nonzero(np.diff(sorted_values) > tolerance))


def _afhp_bin_index(afhp: float, coverage_fraction: float) -> int:
    num_bins = int(1.0 / coverage_fraction)
    afhp = min(max(float(afhp), 0.0), 1.0)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_index = int(np.searchsorted(edges, afhp, side="right") - 1)
    return min(max(bin_index, 0), num_bins - 1)


def image_svdd_probe_decision(
    diagnostics: Dict[str, Any],
    finite_probe_points: List[Dict[str, Any]],
    *,
    coverage_fraction: float,
    high_afhp_threshold: float = DEFAULT_IMAGE_SVDD_HIGH_AFHP_THRESHOLD,
) -> Dict[str, Any]:
    """Decide whether image SVDD should switch from percentile search to raw expansion."""
    if not diagnostics.get("is_image_svdd", False):
        return {"should_expand": False, "reason": "not_image_svdd"}
    if diagnostics.get("max_score") is None:
        return {"should_expand": False, "reason": "missing_id_threshold"}
    if diagnostics.get("is_degenerate", False):
        return {"should_expand": True, "reason": "degenerate_calibration"}
    if not finite_probe_points:
        return {"should_expand": False, "reason": "no_finite_probe_points"}

    tolerance = float(
        diagnostics.get("tolerance")
        or max(
            DEFAULT_SCORE_TOLERANCE_ABS,
            abs(float(diagnostics["max_score"])) * DEFAULT_SCORE_TOLERANCE_REL,
        )
    )
    finite_thresholds = np.asarray(
        [
            point["threshold"]
            for point in finite_probe_points
            if np.isfinite(point["threshold"])
        ],
        dtype=float,
    )
    if finite_thresholds.size == 0:
        return {"should_expand": False, "reason": "no_finite_probe_thresholds"}

    threshold_unique_count = _unique_count(finite_thresholds, tolerance)
    if threshold_unique_count < 2:
        return {
            "should_expand": True,
            "reason": "duplicate_finite_thresholds",
            "threshold_unique_count": threshold_unique_count,
        }

    finite_afhps = np.asarray(
        [point["afhp"] for point in finite_probe_points], dtype=float
    )
    if np.all(finite_afhps >= high_afhp_threshold):
        return {
            "should_expand": True,
            "reason": "all_finite_probes_high_afhp",
            "min_probe_afhp": float(np.min(finite_afhps)),
            "max_probe_afhp": float(np.max(finite_afhps)),
        }

    probe_bins = {
        _afhp_bin_index(point["afhp"], coverage_fraction)
        for point in finite_probe_points
    }
    return {
        "should_expand": False,
        "reason": "probe_percentile_search_healthy",
        "threshold_unique_count": threshold_unique_count,
        "probe_bin_count": len(probe_bins),
        "min_probe_afhp": float(np.min(finite_afhps)),
        "max_probe_afhp": float(np.max(finite_afhps)),
    }


def _log_image_svdd_probe_point(label: str, threshold: float, afhp: float) -> None:
    logging.info(
        "Image SVDD probe %s: threshold=%.8g, level_afhp=%.4f",
        label,
        threshold,
        afhp,
    )


def _augment_sampling_result_with_probe_info(
    result: SamplingResult,
    *,
    probe_info: Dict[str, Any],
    probe_eval_count: int,
) -> SamplingResult:
    info = dict(result.info or {})
    info.update(probe_info)
    return SamplingResult(
        points=result.points,
        coverage_x_max_gap=result.coverage_x_max_gap,
        coverage_y_max_gap=result.coverage_y_max_gap,
        total_evals=result.total_evals + probe_eval_count,
        early_stop_reason=result.early_stop_reason,
        monotonicity_violations_remaining=result.monotonicity_violations_remaining,
        info=info,
    )


class ImageSVDDRawThresholdSampler:
    """Fallback sampler for image SVDD when ID percentile scores collapse.

    The standard sampler searches percentile space. If all ID calibration
    episode-max scores are the same, every interior percentile maps to the same
    threshold. This sampler searches raw thresholds above the ID plateau.
    """

    def __init__(
        self,
        *,
        eval_with_threshold: Callable[[float], Tuple[float, float, Dict[str, Any]]],
        id_threshold: float,
        coverage_fraction: float,
        max_total_evals: int,
        expansion_max_evals: int = DEFAULT_IMAGE_SVDD_EXPANSION_MAX_EVALS,
        expansion_initial_delta_fraction: float = (
            DEFAULT_IMAGE_SVDD_EXPANSION_INITIAL_DELTA_FRACTION
        ),
        strategy_name: str = IMAGE_SVDD_DEGENERATE_STRATEGY,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not np.isfinite(id_threshold):
            raise ValueError(
                "Image SVDD raw-threshold fallback needs a finite ID threshold."
            )
        if not (0.0 < coverage_fraction <= 1.0):
            raise ValueError("coverage_fraction must be in (0, 1].")
        if max_total_evals < 2:
            raise ValueError("max_total_evals must be at least 2.")

        self.eval_with_threshold = eval_with_threshold
        self.id_threshold = float(id_threshold)
        self.coverage_fraction = float(coverage_fraction)
        self.max_total_evals = int(max_total_evals)
        self.expansion_max_evals = int(expansion_max_evals)
        self.expansion_initial_delta_fraction = float(expansion_initial_delta_fraction)
        self.strategy_name = strategy_name
        self.diagnostics = diagnostics or {}

        self.points: List[Dict[str, Any]] = []
        self.seen_thresholds = set()
        self.unfillable_intervals: List[Dict[str, Any]] = []
        self.early_stop_reason: Optional[str] = None
        self.upper_threshold: Optional[float] = None

    def run(self) -> SamplingResult:
        self._evaluate_threshold(self.id_threshold, role="id_threshold")

        next_threshold = float(np.nextafter(self.id_threshold, float("inf")))
        next_point = self._evaluate_threshold(next_threshold, role="id_threshold_next")
        if next_point["afhp"] <= self._low_afhp_target():
            self.upper_threshold = next_threshold
        else:
            self.upper_threshold = self._expand_upper_threshold()

        if self.upper_threshold is None and len(self.points) < self.max_total_evals:
            self._evaluate_threshold(float("inf"), role="upper_extreme")

        self._fill_afhp_bins()
        return self._sampling_result()

    def _low_afhp_target(self) -> float:
        return self.coverage_fraction / 2.0

    def _threshold_key(self, threshold: float) -> Tuple[bool, float]:
        if np.isposinf(threshold):
            return (True, float("inf"))
        if np.isneginf(threshold):
            return (True, float("-inf"))
        return (False, float(threshold))

    def _evaluate_threshold(self, threshold: float, *, role: str) -> Dict[str, Any]:
        key = self._threshold_key(threshold)
        for point in self.points:
            if point["key"] == key:
                return point

        if len(self.points) >= self.max_total_evals:
            self.early_stop_reason = "max_total_evals"
            raise RuntimeError("Image SVDD threshold sampler exceeded max_total_evals.")

        afhp, performance, meta = self.eval_with_threshold(float(threshold))
        metadata = dict(meta or {})
        metadata.update(
            {
                "threshold_strategy": self.strategy_name,
                "degenerate_calibration_detected": bool(
                    self.diagnostics.get("is_degenerate", False)
                ),
                "id_threshold": self.id_threshold,
                "raw_threshold": float(threshold),
                "threshold_role": role,
            }
        )

        point = {
            "threshold": float(threshold),
            "key": key,
            "afhp": float(afhp),
            "performance": float(performance),
            "meta": metadata,
            "order": len(self.points) + 1,
        }
        self.points.append(point)
        self.seen_thresholds.add(key)
        return point

    def _expand_upper_threshold(self) -> Optional[float]:
        delta = max(
            abs(self.id_threshold) * self.expansion_initial_delta_fraction,
            1e-6,
        )
        for expansion_index in range(self.expansion_max_evals):
            if len(self.points) >= self.max_total_evals:
                self.early_stop_reason = "max_total_evals"
                return None
            threshold = self.id_threshold + delta * (2**expansion_index)
            point = self._evaluate_threshold(
                threshold,
                role=f"expansion_{expansion_index + 1}",
            )
            if point["afhp"] <= self._low_afhp_target():
                return float(threshold)

        self.early_stop_reason = "expansion_upper_not_found"
        return None

    def _fill_afhp_bins(self) -> None:
        while not self._coverage_satisfied():
            if len(self.points) >= self.max_total_evals:
                self.early_stop_reason = "max_total_evals"
                return

            interval = self._select_widest_fillable_interval()
            if interval is None:
                if self.early_stop_reason is None:
                    self.early_stop_reason = "unfillable_afhp_intervals"
                return

            left, right = interval
            threshold_left = left["threshold"]
            threshold_right = right["threshold"]
            midpoint = self._midpoint_threshold(threshold_left, threshold_right)
            if midpoint is None:
                self._record_unfillable_interval(left, right, "threshold_precision")
                continue
            if self._threshold_key(midpoint) in self.seen_thresholds:
                self._record_unfillable_interval(left, right, "duplicate_threshold")
                continue

            self._evaluate_threshold(midpoint, role="bisect")

    def _midpoint_threshold(self, left: float, right: float) -> Optional[float]:
        if not (np.isfinite(left) and np.isfinite(right)):
            return None
        midpoint = float((left + right) / 2.0)
        if midpoint == left or midpoint == right:
            return None
        return midpoint

    def _coverage_satisfied(self) -> bool:
        return self._bins_filled() == self._num_bins()

    def _num_bins(self) -> int:
        return int(1.0 / self.coverage_fraction)

    def _bin_edges(self):
        return np.linspace(0.0, 1.0, self._num_bins() + 1)

    def _bin_index(self, afhp: float) -> int:
        afhp = min(max(float(afhp), 0.0), 1.0)
        edges = self._bin_edges()
        bin_index = int(np.searchsorted(edges, afhp, side="right") - 1)
        return min(max(bin_index, 0), self._num_bins() - 1)

    def _filled_bin_indices(self) -> set:
        return {self._bin_index(point["afhp"]) for point in self.points}

    def _bins_filled(self) -> int:
        return len(self._filled_bin_indices())

    def _select_widest_fillable_interval(self):
        # Tie-break ascending AFHP by descending threshold so the
        # threshold-closest pair across an AFHP step ends up adjacent.
        # Assumes AFHP is monotone non-increasing in threshold, which holds
        # for the image-SVDD raw-threshold path above the ID plateau.
        ordered = sorted(
            self.points, key=lambda point: (point["afhp"], -point["threshold"])
        )
        best_interval = None
        best_gap = -1.0
        for left, right in zip(ordered[:-1], ordered[1:]):
            gap = right["afhp"] - left["afhp"]
            if gap <= self.coverage_fraction:
                continue
            if not self._interval_has_unfilled_bin(left["afhp"], right["afhp"]):
                continue
            if self._interval_recorded_unfillable(left, right):
                continue
            if not (np.isfinite(left["threshold"]) and np.isfinite(right["threshold"])):
                self._record_unfillable_interval(left, right, "infinite_threshold")
                continue
            if gap > best_gap:
                best_gap = gap
                best_interval = (left, right)
        return best_interval

    def _interval_has_unfilled_bin(self, afhp_left: float, afhp_right: float) -> bool:
        filled = self._filled_bin_indices()
        edges = self._bin_edges()
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            if afhp_left < high and low < afhp_right and bin_index not in filled:
                return True
        return False

    def _interval_recorded_unfillable(self, left, right) -> bool:
        keys = {left["key"], right["key"]}
        for interval in self.unfillable_intervals:
            if interval["keys"] == keys:
                return True
        return False

    def _record_unfillable_interval(self, left, right, reason: str) -> None:
        if self._interval_recorded_unfillable(left, right):
            return
        self.unfillable_intervals.append(
            {
                "keys": {left["key"], right["key"]},
                "threshold_low_afhp": left["threshold"],
                "threshold_high_afhp": right["threshold"],
                "afhp_low": left["afhp"],
                "afhp_high": right["afhp"],
                "reason": reason,
            }
        )

    def _coverage_gap(self) -> float:
        values = sorted(min(max(point["afhp"], 0.0), 1.0) for point in self.points)
        boundaries = [0.0, *values, 1.0]
        return max(
            boundaries[index + 1] - boundaries[index]
            for index in range(len(boundaries) - 1)
        )

    def _sampling_result(self) -> SamplingResult:
        upper_threshold = self.upper_threshold
        if upper_threshold is None:
            finite_thresholds = [
                point["threshold"]
                for point in self.points
                if np.isfinite(point["threshold"])
            ]
            upper_threshold = (
                max(finite_thresholds) if finite_thresholds else self.id_threshold
            )

        threshold_span = max(float(upper_threshold) - self.id_threshold, 0.0)
        curve_points = []
        for point in self.points:
            threshold = point["threshold"]
            if np.isposinf(threshold):
                desired_percentile = 1.0
            elif threshold_span > 0.0:
                desired_percentile = (threshold - self.id_threshold) / threshold_span
                desired_percentile = min(max(float(desired_percentile), 0.0), 1.0)
            else:
                desired_percentile = 0.0

            point["meta"]["expansion_upper_threshold"] = upper_threshold
            curve_points.append(
                CurvePoint(
                    desired_percentile=desired_percentile,
                    afhp=point["afhp"],
                    performance=point["performance"],
                    repeats_used=1,
                    order=point["order"],
                    meta=point["meta"],
                )
            )

        unfillable_intervals = [
            {key: value for key, value in interval.items() if key != "keys"}
            for interval in self.unfillable_intervals
        ]
        info = {
            "threshold_strategy": self.strategy_name,
            "degenerate_calibration_detected": bool(
                self.diagnostics.get("is_degenerate", False)
            ),
            "id_threshold": self.id_threshold,
            "expansion_upper_threshold": upper_threshold,
            "unfillable_afhp_intervals": unfillable_intervals,
            "bins_filled": self._bins_filled(),
            "total_bins": self._num_bins(),
            "coverage_percentage": 100.0 * self._bins_filled() / self._num_bins(),
            "early_stop_reason": self.early_stop_reason,
            "calibration_diagnostics": self.diagnostics,
        }
        return SamplingResult(
            points=curve_points,
            coverage_x_max_gap=self._coverage_gap(),
            coverage_y_max_gap=0.0,
            total_evals=len(self.points),
            early_stop_reason=self.early_stop_reason,
            monotonicity_violations_remaining=False,
            info=info,
        )


class ImageSVDDProbeSampler:
    """Probe percentile-space behavior before choosing the image-SVDD sampler path."""

    def __init__(
        self,
        *,
        diagnostics: Dict[str, Any],
        coverage_fraction: float,
        eval_at_percentile: Callable[[float], Tuple[float, float, Dict[str, Any]]],
        eval_at_lower_extreme: Callable[[], Tuple[float, float, Dict[str, Any]]],
        eval_at_upper_extreme: Callable[[], Tuple[float, float, Dict[str, Any]]],
        eval_with_threshold: Callable[[float], Tuple[float, float, Dict[str, Any]]],
        binary_sampler_factory: Callable[[], Any],
        max_total_evals: int,
        expansion_max_evals: int,
        expansion_initial_delta_fraction: float,
        strategy_name: str,
        probe_percentiles: Tuple[float, ...] = DEFAULT_IMAGE_SVDD_PROBE_PERCENTILES,
    ) -> None:
        self.diagnostics = diagnostics
        self.coverage_fraction = coverage_fraction
        self.eval_at_percentile = eval_at_percentile
        self.eval_at_lower_extreme = eval_at_lower_extreme
        self.eval_at_upper_extreme = eval_at_upper_extreme
        self.eval_with_threshold = eval_with_threshold
        self.binary_sampler_factory = binary_sampler_factory
        self.max_total_evals = max_total_evals
        self.expansion_max_evals = expansion_max_evals
        self.expansion_initial_delta_fraction = expansion_initial_delta_fraction
        self.strategy_name = strategy_name
        self.probe_percentiles = tuple(probe_percentiles)

    def run(self) -> SamplingResult:
        logging.info(
            "Image SVDD percentile probe starting with percentiles=%s",
            self.probe_percentiles,
        )
        probe_points = []

        lower_afhp, _, lower_meta = self.eval_at_lower_extreme()
        lower_threshold = float(lower_meta["threshold"])
        _log_image_svdd_probe_point("lower_extreme", lower_threshold, lower_afhp)
        probe_points.append(
            {"label": "lower_extreme", "threshold": lower_threshold, "afhp": lower_afhp}
        )

        upper_afhp, _, upper_meta = self.eval_at_upper_extreme()
        upper_threshold = float(upper_meta["threshold"])
        _log_image_svdd_probe_point("upper_extreme", upper_threshold, upper_afhp)
        probe_points.append(
            {"label": "upper_extreme", "threshold": upper_threshold, "afhp": upper_afhp}
        )

        finite_probe_points = []
        for percentile in self.probe_percentiles:
            afhp, _, meta = self.eval_at_percentile(percentile)
            threshold = float(meta["threshold"])
            label = f"p={percentile:.2f}"
            _log_image_svdd_probe_point(label, threshold, afhp)
            point = {
                "label": label,
                "percentile": percentile,
                "threshold": threshold,
                "afhp": afhp,
            }
            probe_points.append(point)
            if np.isfinite(threshold):
                finite_probe_points.append(point)

        decision = image_svdd_probe_decision(
            self.diagnostics,
            finite_probe_points,
            coverage_fraction=self.coverage_fraction,
        )
        probe_info = {
            "image_svdd_probe_triggered": True,
            "image_svdd_probe_points": probe_points,
            "image_svdd_probe_reason": decision["reason"],
        }
        probe_info.update(
            {
                key: value
                for key, value in decision.items()
                if key not in {"should_expand", "reason"}
            }
        )

        if decision["should_expand"]:
            id_threshold = float(self.diagnostics["max_score"])
            logging.warning(
                "Image SVDD percentile search collapsed; switching to raw threshold expansion above ID threshold %.8g. Reason=%s Diagnostics=%s",
                id_threshold,
                decision["reason"],
                self.diagnostics,
            )
            result = ImageSVDDRawThresholdSampler(
                eval_with_threshold=self.eval_with_threshold,
                id_threshold=id_threshold,
                coverage_fraction=self.coverage_fraction,
                max_total_evals=max(self.max_total_evals - len(probe_points), 2),
                expansion_max_evals=self.expansion_max_evals,
                expansion_initial_delta_fraction=self.expansion_initial_delta_fraction,
                strategy_name=self.strategy_name,
                diagnostics=self.diagnostics,
            ).run()
            return _augment_sampling_result_with_probe_info(
                result,
                probe_info=probe_info,
                probe_eval_count=len(probe_points),
            )

        logging.info(
            "Image SVDD percentile probe looks healthy; continuing with percentile sampler. Decision=%s",
            decision,
        )
        result = self.binary_sampler_factory().run()
        return _augment_sampling_result_with_probe_info(
            result,
            probe_info=probe_info,
            probe_eval_count=len(probe_points),
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
    image_svdd_degenerate_strategy: str = IMAGE_SVDD_DEGENERATE_STRATEGY,
    image_svdd_expansion_max_evals: int = DEFAULT_IMAGE_SVDD_EXPANSION_MAX_EVALS,
    image_svdd_expansion_initial_delta_fraction: float = (
        DEFAULT_IMAGE_SVDD_EXPANSION_INITIAL_DELTA_FRACTION
    ),
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

    image_svdd_diagnostics = image_svdd_calibration_diagnostics(policy)
    logging.info(
        "Sampler dispatch: policy_type=%s, clf_name=%s, feature_type=%s, "
        "image_svdd_degenerate_strategy=%r, image_svdd_diagnostics=%s",
        type(policy).__name__,
        getattr(policy, "clf_name", None),
        getattr(policy, "feature_type", None),
        image_svdd_degenerate_strategy,
        image_svdd_diagnostics,
    )
    if (
        image_svdd_degenerate_strategy == IMAGE_SVDD_DEGENERATE_STRATEGY
        and image_svdd_diagnostics["is_image_svdd"]
        and image_svdd_diagnostics["max_score"] is not None
    ):
        logging.info(
            "Sampler dispatch: selecting ImageSVDDProbeSampler "
            "(strategy=%s, max_score=%.8g, unique_count=%d, is_degenerate=%s)",
            image_svdd_degenerate_strategy,
            image_svdd_diagnostics["max_score"],
            image_svdd_diagnostics.get("unique_count", -1),
            image_svdd_diagnostics["is_degenerate"],
        )
        return ImageSVDDProbeSampler(
            diagnostics=image_svdd_diagnostics,
            coverage_fraction=coverage_fraction,
            eval_at_percentile=eval_at_percentile,
            eval_at_lower_extreme=eval_at_lower_extreme,
            eval_at_upper_extreme=eval_at_upper_extreme,
            eval_with_threshold=_eval_with_threshold,
            binary_sampler_factory=lambda: BinarySearchSampler(
                eval_at_percentile=eval_at_percentile,
                eval_at_lower_extreme=eval_at_lower_extreme,
                eval_at_upper_extreme=eval_at_upper_extreme,
                num_bins=num_bins,
                output_range=(0.0, 1.0),
            ),
            max_total_evals=max_total_evals,
            expansion_max_evals=image_svdd_expansion_max_evals,
            expansion_initial_delta_fraction=(
                image_svdd_expansion_initial_delta_fraction
            ),
            strategy_name=image_svdd_degenerate_strategy,
        )

    # Use WaitPolicyAwareSampler if we have a WaitPolicy, otherwise use regular BinarySearchSampler
    if isinstance(policy, WaitPolicy):
        logging.info("Sampler dispatch: selecting WaitPolicyAwareSampler")
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
        logging.info(
            "Sampler dispatch: selecting BinarySearchSampler "
            "(image_svdd probe NOT triggered)"
        )
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
    if isinstance(policy, OODPolicy) or isinstance(policy, ThresholdPolicy):
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

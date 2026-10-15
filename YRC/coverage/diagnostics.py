"""Diagnostics for finite-threshold level-AFHP search ranges."""

import math
from typing import Any, Dict, List, Optional

import numpy as np


_WILSON_Z_95 = 1.959963984540054
_METHOD = "empirical_episode_max_strict_gt"
_UNCERTAINTY_NOTE = (
    "Wilson intervals approximate episode-sampling uncertainty at the two "
    "finite threshold endpoints; they are not formal bounds across repeated "
    "stochastic rollouts."
)


def _wilson_interval(successes: int, sample_count: int) -> Dict[str, float]:
    """Compute a two-sided 95% Wilson interval for a binomial proportion."""
    proportion = successes / sample_count
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / sample_count
    center = (proportion + z_squared / (2.0 * sample_count)) / denominator
    half_width = (
        _WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_count
            + z_squared / (4.0 * sample_count**2)
        )
        / denominator
    )
    return {
        "lower": float(max(0.0, center - half_width)),
        "upper": float(min(1.0, center + half_width)),
    }


def _bins_outside_interval(
    interval_min: float, interval_max: float, num_bins: int
) -> List[int]:
    """Return whole bins disjoint from a closed AFHP interval.

    Coverage bins use left-inclusive, right-exclusive edges (with AFHP 1.0
    clamped into the final bin), as does the adaptive coverage sampler. A bin
    that only partly lies beyond an interval is kept out of this list.
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    outside = []
    for index in range(num_bins):
        bin_min = float(edges[index])
        bin_max = float(edges[index + 1])
        # The sampler clamps AFHP 1.0 into the final bin, so that endpoint is
        # included even though the other bins are right-exclusive.
        final_bin_contains_one = index == num_bins - 1 and interval_min == 1.0
        wholly_below = bin_max <= interval_min and not final_bin_contains_one
        wholly_above = bin_min > interval_max
        if wholly_below or wholly_above:
            outside.append(index)
    return outside


def _empty_diagnostics(
    *,
    status: str,
    total_count: Optional[int],
    sample_count: int,
    invalid_count: Optional[int],
    finite_min: Optional[float],
    finite_max: Optional[float],
) -> Dict[str, Any]:
    return {
        "status": status,
        "total_count": total_count,
        "sample_count": sample_count,
        "invalid_count": invalid_count,
        "finite_min": finite_min,
        "finite_max": finite_max,
        "estimated_min_afhp": None,
        "estimated_max_afhp": None,
        "min_afhp_wilson_95": None,
        "max_afhp_wilson_95": None,
        "reachable_afhp_interval": None,
        "uncertainty_afhp_interval": None,
        "range_limited_suspected": False,
        "requested_bins_beyond_range": [],
        "empirical_bins_beyond_range": [],
        "method": _METHOD,
        "uncertainty_note": _UNCERTAINTY_NOTE,
        "advisory_only": True,
        "limitation_scope": "finite_threshold_range_only",
    }


def level_afhp_range_diagnostics(
    episode_max_scores: Any,
    finite_min: Optional[float],
    finite_max: Optional[float],
    num_bins: int,
) -> Dict[str, Any]:
    """Estimate the level-AFHP range covered by finite score thresholds.

    ``episode_max_scores`` must contain one maximum of the actual decision
    score per episode, collected while the novice runs without help. The score
    must include any smoothing used by the policy. With the threshold policy's
    strict ``score > threshold`` decision, the empirical AFHP at threshold
    ``t`` is ``mean(episode_max_scores > t)``. Thus the finite threshold range
    has its minimum empirical AFHP at ``finite_max`` and its maximum at
    ``finite_min``.

    Estimates are returned only when every submitted episode score is finite.
    Missing, malformed, or non-finite data cannot support a conservative
    range warning, so their rate estimates and suspected-bin lists are left
    unavailable. ``requested_bins_beyond_range`` includes only complete bins
    outside the endpoint Wilson uncertainty band; ``empirical_bins_beyond_range``
    separately shows bins outside the point-estimate interval. Both describe
    the finite-threshold range, not bins left unfilled by the overall sampler.

    Args:
        episode_max_scores: One finite decision-score maximum per no-help
            episode, or ``None`` when the scores are unavailable.
        finite_min: Lowest finite threshold the search may evaluate.
        finite_max: Highest finite threshold the search may evaluate.
        num_bins: Number of equal-width AFHP bins used by the sampler.

    Returns:
        A plain-serializable dictionary containing endpoint estimates,
        approximate Wilson intervals, and advisory suspected bins.
    """
    try:
        threshold_min = float(finite_min)
        threshold_max = float(finite_max)
    except (TypeError, ValueError, OverflowError):
        return _empty_diagnostics(
            status="invalid_range",
            total_count=None,
            sample_count=0,
            invalid_count=None,
            finite_min=None,
            finite_max=None,
        )
    if not (math.isfinite(threshold_min) and math.isfinite(threshold_max)):
        return _empty_diagnostics(
            status="invalid_range",
            total_count=None,
            sample_count=0,
            invalid_count=None,
            finite_min=None,
            finite_max=None,
        )
    if threshold_min > threshold_max:
        return _empty_diagnostics(
            status="invalid_range",
            total_count=None,
            sample_count=0,
            invalid_count=None,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )
    if isinstance(num_bins, bool) or not isinstance(num_bins, (int, np.integer)):
        raise ValueError("num_bins must be a positive integer")
    num_bins = int(num_bins)
    if num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    if episode_max_scores is None:
        return _empty_diagnostics(
            status="missing_scores",
            total_count=None,
            sample_count=0,
            invalid_count=None,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )

    try:
        raw_scores = np.asarray(episode_max_scores)
    except (TypeError, ValueError):
        return _empty_diagnostics(
            status="invalid_scores",
            total_count=None,
            sample_count=0,
            invalid_count=None,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )

    total_count = int(raw_scores.size)
    if raw_scores.ndim != 1:
        return _empty_diagnostics(
            status="unsupported_scores",
            total_count=total_count,
            sample_count=0,
            invalid_count=total_count,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )
    if total_count == 0:
        return _empty_diagnostics(
            status="no_scores",
            total_count=0,
            sample_count=0,
            invalid_count=0,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )

    try:
        scores = np.asarray(raw_scores, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _empty_diagnostics(
            status="invalid_scores",
            total_count=total_count,
            sample_count=0,
            invalid_count=total_count,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )

    finite_mask = np.isfinite(scores)
    sample_count = int(np.count_nonzero(finite_mask))
    invalid_count = total_count - sample_count
    if invalid_count:
        return _empty_diagnostics(
            status="nonfinite_scores",
            total_count=total_count,
            sample_count=sample_count,
            invalid_count=invalid_count,
            finite_min=threshold_min,
            finite_max=threshold_max,
        )

    min_successes = int(np.count_nonzero(scores > threshold_max))
    max_successes = int(np.count_nonzero(scores > threshold_min))
    estimated_min_afhp = float(min_successes / sample_count)
    estimated_max_afhp = float(max_successes / sample_count)
    min_interval = _wilson_interval(min_successes, sample_count)
    max_interval = _wilson_interval(max_successes, sample_count)
    empirical_interval = {
        "min": estimated_min_afhp,
        "max": estimated_max_afhp,
    }
    uncertainty_interval = {
        "min": min_interval["lower"],
        "max": max_interval["upper"],
    }
    suspected_bins = _bins_outside_interval(
        uncertainty_interval["min"], uncertainty_interval["max"], num_bins
    )
    empirical_bins = _bins_outside_interval(
        empirical_interval["min"], empirical_interval["max"], num_bins
    )

    return {
        "status": "ok",
        "total_count": total_count,
        "sample_count": sample_count,
        "invalid_count": 0,
        "finite_min": threshold_min,
        "finite_max": threshold_max,
        "estimated_min_afhp": estimated_min_afhp,
        "estimated_max_afhp": estimated_max_afhp,
        "min_afhp_wilson_95": min_interval,
        "max_afhp_wilson_95": max_interval,
        "reachable_afhp_interval": empirical_interval,
        "uncertainty_afhp_interval": uncertainty_interval,
        "range_limited_suspected": bool(suspected_bins),
        "requested_bins_beyond_range": suspected_bins,
        "empirical_bins_beyond_range": empirical_bins,
        "method": _METHOD,
        "uncertainty_note": _UNCERTAINTY_NOTE,
        "advisory_only": True,
        "limitation_scope": "finite_threshold_range_only",
    }

import numpy as np
import pytest

from YRC.coverage.diagnostics import level_afhp_range_diagnostics


def test_detects_synthetic_82_percent_cliff_with_uncertainty_band():
    scores = np.concatenate([np.full(8200, 1.5), np.full(1800, -1.0)])

    result = level_afhp_range_diagnostics(scores, 0.0, 1.0, 20)

    assert result["status"] == "ok"
    assert result["total_count"] == 10_000
    assert result["sample_count"] == 10_000
    assert result["invalid_count"] == 0
    assert result["estimated_min_afhp"] == pytest.approx(0.82)
    assert result["estimated_max_afhp"] == pytest.approx(0.82)
    assert result["reachable_afhp_interval"] == pytest.approx(
        {"min": 0.82, "max": 0.82}
    )
    assert result["range_limited_suspected"] is True
    assert result["requested_bins_beyond_range"] == list(range(16)) + [17, 18, 19]
    assert result["empirical_bins_beyond_range"] == list(range(16)) + [17, 18, 19]
    assert result["limitation_scope"] == "finite_threshold_range_only"
    assert result["advisory_only"] is True


def test_small_sample_reports_wide_wilson_uncertainty():
    result = level_afhp_range_diagnostics([2.0, 0.5], 1.0, 1.5, 20)

    assert result["estimated_min_afhp"] == pytest.approx(0.5)
    assert result["estimated_max_afhp"] == pytest.approx(0.5)
    min_interval = result["min_afhp_wilson_95"]
    max_interval = result["max_afhp_wilson_95"]
    assert min_interval["lower"] == pytest.approx(0.0945, abs=0.001)
    assert min_interval["upper"] == pytest.approx(0.9055, abs=0.001)
    assert max_interval == pytest.approx(min_interval)
    assert result["uncertainty_afhp_interval"]["min"] == min_interval["lower"]
    assert result["uncertainty_afhp_interval"]["max"] == max_interval["upper"]
    assert result["requested_bins_beyond_range"] == [0, 19]
    assert "approximate episode-sampling uncertainty" in result["uncertainty_note"]


def test_strict_greater_than_excludes_scores_tied_at_finite_threshold():
    result = level_afhp_range_diagnostics([1.0, 1.0, 1.0], 1.0, 1.0, 10)

    assert result["estimated_min_afhp"] == 0.0
    assert result["estimated_max_afhp"] == 0.0
    assert result["reachable_afhp_interval"] == {"min": 0.0, "max": 0.0}
    assert result["range_limited_suspected"] is True


@pytest.mark.parametrize(
    "scores, expected_rate, expected_outside_bins",
    [
        ([2.0, 2.0], 1.0, list(range(9))),
        ([-2.0, -2.0], 0.0, list(range(1, 10))),
    ],
)
def test_exact_zero_and_one_afhp_keep_sampler_endpoint_bins(
    scores, expected_rate, expected_outside_bins
):
    result = level_afhp_range_diagnostics(scores, 0.0, 1.0, 10)

    assert result["estimated_min_afhp"] == expected_rate
    assert result["estimated_max_afhp"] == expected_rate
    assert result["empirical_bins_beyond_range"] == expected_outside_bins


@pytest.mark.parametrize(
    "scores, expected_status, total_count, sample_count, invalid_count",
    [
        (None, "missing_scores", None, 0, None),
        ([], "no_scores", 0, 0, 0),
        ([np.nan, np.inf], "nonfinite_scores", 2, 0, 2),
        ([0.5, np.nan, 1.5], "nonfinite_scores", 3, 2, 1),
        (["not-a-score"], "invalid_scores", 1, 0, 1),
        ([[0.5, 1.5]], "unsupported_scores", 2, 0, 2),
    ],
)
def test_unavailable_or_invalid_scores_never_raise_range_suspicion(
    scores, expected_status, total_count, sample_count, invalid_count
):
    result = level_afhp_range_diagnostics(scores, 0.0, 1.0, 20)

    assert result["status"] == expected_status
    assert result["total_count"] == total_count
    assert result["sample_count"] == sample_count
    assert result["invalid_count"] == invalid_count
    assert result["estimated_min_afhp"] is None
    assert result["estimated_max_afhp"] is None
    assert result["range_limited_suspected"] is False
    assert result["requested_bins_beyond_range"] == []
    assert result["empirical_bins_beyond_range"] == []


@pytest.mark.parametrize(
    "finite_min, finite_max, expected_min, expected_max",
    [
        (None, 1.0, None, None),
        (np.nan, 1.0, None, None),
        (0.0, np.inf, None, None),
        (1.0, 0.0, 1.0, 0.0),
    ],
)
def test_invalid_finite_range_is_unavailable_not_suspected(
    finite_min, finite_max, expected_min, expected_max
):
    result = level_afhp_range_diagnostics([0.5, 1.5], finite_min, finite_max, 20)

    assert result["status"] == "invalid_range"
    assert result["finite_min"] == expected_min
    assert result["finite_max"] == expected_max
    assert result["range_limited_suspected"] is False
    assert result["requested_bins_beyond_range"] == []
    assert result["estimated_min_afhp"] is None


@pytest.mark.parametrize("num_bins", [0, 2.5, True])
def test_invalid_bin_count_is_rejected(num_bins):
    with pytest.raises(ValueError, match="num_bins"):
        level_afhp_range_diagnostics([0.5, 1.5], 0.0, 1.0, num_bins)

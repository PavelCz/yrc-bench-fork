from types import SimpleNamespace

import numpy as np
import pytest

from YRC.coverage import coverage_search
from YRC.coverage.coverage_search import (
    LevelAFHPThresholdSampler,
    create_level_afhp_threshold_sampler,
)
from YRC.policies.heuristic import WaitPolicy
from YRC.policies.threshold import ThresholdPolicy


def _search_diagnostics():
    return {
        "status": "in_progress",
        "requested_threshold_count": 0,
        "unique_threshold_count": 0,
        "reused_threshold_request_count": 0,
        "reused_threshold_requests": [],
        "exhausted_intervals": [],
        "unfilled_bin_indices": [],
    }


def _bare_sampler(threshold_for_input, afhp_for_threshold):
    evaluated_thresholds = []

    def effective_threshold_for_input(percentile):
        if abs(percentile - 0.0) < 1e-9:
            return float("inf")
        if abs(percentile - 1.0) < 1e-9:
            return float("-inf")
        return threshold_for_input(percentile)

    def evaluate_threshold(threshold):
        threshold = float(threshold)
        evaluated_thresholds.append(threshold)
        return afhp_for_threshold(threshold), threshold, {"threshold": threshold}

    def evaluate_percentile(percentile):
        return evaluate_threshold(threshold_for_input(float(percentile)))

    sampler = LevelAFHPThresholdSampler(
        eval_at_percentile=evaluate_percentile,
        eval_at_lower_extreme=lambda: evaluate_threshold(float("inf")),
        eval_at_upper_extreme=lambda: evaluate_threshold(float("-inf")),
        threshold_for_input=effective_threshold_for_input,
        num_bins=20,
        diagnostics=_search_diagnostics(),
        threshold_mapping_is_monotone=True,
    )
    return sampler, evaluated_thresholds


def test_constant_threshold_interval_exhausts_without_extra_evaluation():
    sampler, evaluated_thresholds = _bare_sampler(
        lambda percentile: 1.0,
        lambda threshold: 0.2,
    )

    evaluations = sampler.binary_search_fill(0.25, 0.75, 0, 19)

    assert evaluations == 0
    assert evaluated_thresholds == []
    interval = sampler.diagnostics["exhausted_intervals"][0]
    assert interval["reason"] == "constant_threshold_mapping"
    assert interval["unresolved_bin_indices"] == list(range(1, 19))

    first = sampler.evaluate_at_input(0.25)
    repeated = sampler.evaluate_at_input(0.75)
    assert repeated is first
    assert len(sampler.all_samples) == 1
    assert sampler.total_evals == 1
    assert evaluated_thresholds == [1.0]
    assert sampler.diagnostics["reused_threshold_request_count"] == 1


def test_failed_evaluation_rolls_back_unique_threshold_count():
    def fail_evaluation(percentile):
        raise RuntimeError("synthetic evaluator failure")

    sampler = LevelAFHPThresholdSampler(
        eval_at_percentile=fail_evaluation,
        eval_at_lower_extreme=fail_evaluation,
        eval_at_upper_extreme=fail_evaluation,
        threshold_for_input=lambda percentile: 0.5,
        num_bins=20,
        diagnostics=_search_diagnostics(),
        threshold_mapping_is_monotone=True,
    )

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        sampler.evaluate_at_input(0.5)

    assert sampler.diagnostics["requested_threshold_count"] == 1
    assert sampler.diagnostics["unique_threshold_count"] == 0
    assert sampler.total_evals == 0


def test_repeated_midpoint_does_not_discard_a_mixed_interval():
    def threshold_for_input(percentile):
        if percentile == 0.0:
            return float("inf")
        if percentile == 1.0:
            return float("-inf")
        return 2.0 if percentile < 0.5 else 1.0

    sampler, evaluated_thresholds = _bare_sampler(
        threshold_for_input,
        lambda threshold: 0.0 if threshold >= 2.0 else 1.0,
    )
    left = sampler.evaluate_at_input(0.25)
    right = sampler.evaluate_at_input(0.75)
    sampler.bin_samples[0] = left
    sampler.bin_samples[19] = right

    sampler.binary_search_fill(0.25, 0.75, 0, 19)

    assert sampler.diagnostics["reused_threshold_request_count"] > 0
    assert evaluated_thresholds == [2.0, 1.0]
    assert not any(
        interval["input_interval"] == [0.25, 0.75]
        for interval in sampler.diagnostics["exhausted_intervals"]
    )
    assert any(
        interval["reason"] == "input_precision_exhausted"
        for interval in sampler.diagnostics["exhausted_intervals"]
    )
    assert sampler.diagnostics["requested_threshold_count"] > 2


def test_afhp_plateau_does_not_stop_search_before_later_movement():
    def threshold_for_input(percentile):
        if percentile == 0.0:
            return float("inf")
        if percentile == 1.0:
            return float("-inf")
        return 1.0 - percentile

    def afhp_for_threshold(threshold):
        if np.isposinf(threshold):
            return 0.0
        if np.isneginf(threshold):
            return 1.0
        if threshold > 0.75:
            return 0.2
        if threshold > 0.25:
            return 0.5
        return 0.8

    sampler, evaluated_thresholds = _bare_sampler(
        threshold_for_input, afhp_for_threshold
    )

    result = sampler.run()
    finite_points = [
        point for point in result.points if np.isfinite(point.meta["threshold"])
    ]

    assert len(evaluated_thresholds) == result.total_evals
    assert any(point.afhp == 0.5 for point in finite_points)
    assert any(point.afhp == 0.8 for point in finite_points)
    assert any(
        first.afhp == second.afhp
        and first.meta["threshold"] != second.meta["threshold"]
        for first in finite_points
        for second in finite_points
    )


class FakeThresholdEvaluator:
    def __init__(self, episode_scores, no_help_episode_max_scores):
        self.episode_scores = np.asarray(episode_scores, dtype=float)
        self.no_help_episode_max_scores = np.asarray(
            no_help_episode_max_scores, dtype=float
        )
        self.thresholds = []

    def eval(self, policy, envs, splits, **kwargs):
        threshold = float(kwargs["threshold"])
        self.thresholds.append(threshold)
        level_ood_pred = self.episode_scores > threshold
        return {
            splits[0]: {
                "level_ood_pred": level_ood_pred,
                "action_1_frac": float(np.mean(level_ood_pred)),
                "env_return_mean": float(np.mean(level_ood_pred)),
                "episode_max_scores": self.no_help_episode_max_scores.tolist(),
            }
        }


def test_no_help_range_warning_is_saved_with_first_and_final_diagnostics(monkeypatch):
    monkeypatch.setattr(
        "YRC.coverage.coverage_search.update_policy_params", lambda *args: None
    )
    policy = ThresholdPolicy.__new__(ThresholdPolicy)
    policy.args = SimpleNamespace(metric="max_logit")
    policy.params = {"threshold": 0.0}
    policy._train_episode_max_scores = np.asarray([0.0, 0.5, 1.0])
    no_help_scores = np.concatenate([np.full(82, 1.5), np.linspace(-30.0, -1.0, 18)])
    evaluator = FakeThresholdEvaluator(no_help_scores, no_help_scores)
    checkpoints = []
    warnings = []

    def record_warning(message, *args):
        warnings.append(message % args if args else message)

    monkeypatch.setattr(coverage_search.logging, "warning", record_warning)

    sampler = create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.05,
        on_evaluation=lambda percentile, afhp, performance, meta: checkpoints.append(
            (percentile, afhp, performance, meta)
        ),
    )

    result = sampler.run()

    first_checkpoint = checkpoints[0]
    assert np.isposinf(first_checkpoint[3]["threshold"])
    range_diagnostics = first_checkpoint[3]["level_afhp_range_diagnostics"]
    assert range_diagnostics["status"] == "ok"
    assert range_diagnostics["estimated_min_afhp"] == 0.82
    assert range_diagnostics["estimated_max_afhp"] == 0.82
    assert range_diagnostics["range_limited_suspected"] is True
    assert (
        first_checkpoint[3]["level_afhp_search_diagnostics"]["unique_threshold_count"]
        == 1
    )
    assert any("82.0%" in warning for warning in warnings)

    final_range_diagnostics = result.info["level_afhp_range_diagnostics"]
    assert final_range_diagnostics["requested_bins_beyond_range"]
    search_diagnostics = result.info["level_afhp_search_diagnostics"]
    assert search_diagnostics["unique_threshold_count"] == result.total_evals
    assert search_diagnostics["requested_threshold_count"] >= result.total_evals
    assert len(evaluator.thresholds) == result.total_evals


def test_non_threshold_policy_is_not_switched_to_degeneracy_sampler(monkeypatch):
    monkeypatch.setattr(
        "YRC.coverage.coverage_search.update_policy_params", lambda *args: None
    )
    policy = SimpleNamespace()
    evaluator = FakeThresholdEvaluator([0.0, 1.0], [0.0, 1.0])

    sampler = create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.5,
    )

    from acs import BinarySearchSampler

    assert isinstance(sampler, BinarySearchSampler)
    assert not isinstance(sampler, LevelAFHPThresholdSampler)
    point = sampler.evaluate_at_input(0.0)
    assert "level_afhp_range_diagnostics" not in point.meta
    assert "level_afhp_search_diagnostics" not in point.meta


def test_wait_policy_keeps_its_existing_sampler(monkeypatch):
    monkeypatch.setattr(
        "YRC.coverage.coverage_search.update_policy_params", lambda *args: None
    )
    wait_policy = WaitPolicy.__new__(WaitPolicy)
    wait_policy.max_episode_length = 100
    evaluator = FakeThresholdEvaluator([0.0, 1.0], [0.0, 1.0])

    sampler = create_level_afhp_threshold_sampler(
        policy=wait_policy,
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.5,
    )

    from acs.wait_policy_sampler import WaitPolicyAwareSampler

    assert isinstance(sampler, WaitPolicyAwareSampler)


def test_instance_percentile_override_disables_range_and_constant_map_claims(
    monkeypatch,
):
    monkeypatch.setattr(
        "YRC.coverage.coverage_search.update_policy_params", lambda *args: None
    )
    policy = ThresholdPolicy.__new__(ThresholdPolicy)
    policy.args = SimpleNamespace(metric="max_logit")
    policy.params = {"threshold": 0.0}
    policy._train_episode_max_scores = np.asarray([0.0, 0.5, 1.0])
    policy.train_percentile_level = lambda percentile: 0.5
    evaluator = FakeThresholdEvaluator([0.0, 1.0], [0.0, 1.0])

    sampler = create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.5,
    )
    no_help = sampler.evaluate_at_input(0.0)

    assert sampler.threshold_mapping_is_monotone is False
    assert (
        no_help.meta["level_afhp_range_diagnostics"]["status"]
        == "unsupported_threshold_mapping"
    )

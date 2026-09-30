import sys
from types import SimpleNamespace

import numpy as np
import pytest

import flags
from YRC.coverage import coverage_search
from YRC.policies.threshold import ThresholdPolicy


class FakeEvaluator:
    def __init__(self, episode_scores, *, max_calls=100):
        self.episode_scores = np.asarray(episode_scores, dtype=float)
        self.thresholds = []
        self.max_calls = max_calls

    def eval(self, policy, envs, splits, **kwargs):
        if len(self.thresholds) >= self.max_calls:
            raise AssertionError("sampler exceeded synthetic evaluation cap")
        threshold = float(kwargs["threshold"])
        self.thresholds.append(threshold)
        split = splits[0]
        level_ood_pred = self.episode_scores > threshold
        return {
            split: {
                "level_ood_pred": level_ood_pred,
                "action_1_frac": float(np.mean(level_ood_pred)),
                "env_return_mean": float(np.mean(level_ood_pred)),
            }
        }


def make_threshold_policy(metric="max_logit", calibration_scores=None):
    policy = ThresholdPolicy.__new__(ThresholdPolicy)
    policy.args = SimpleNamespace(metric=metric)
    if calibration_scores is not None:
        policy._train_episode_max_scores = np.asarray(calibration_scores, dtype=float)
    else:
        policy._train_episode_max_scores = None
    policy.params = {"threshold": 0.0}
    return policy


def make_sampler(policy, evaluator, *, level_threshold_min=None):
    return coverage_search.create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.05,
        level_threshold_min=level_threshold_min,
    )


def test_default_percentile_mapping_and_infinite_endpoints_are_unchanged(
    monkeypatch,
):
    monkeypatch.setattr(coverage_search, "update_policy_params", lambda *args: None)
    policy = make_threshold_policy(calibration_scores=[0.0, 0.5, 1.0])
    sampler = make_sampler(policy, FakeEvaluator([0.25, 1.5]))

    _, _, point = sampler.eval_at_percentile(0.75)
    _, _, lower = sampler.eval_at_lower_extreme()
    _, _, upper = sampler.eval_at_upper_extreme()

    assert point["threshold"] == pytest.approx(0.25)
    assert "level_threshold_mapping" not in point
    assert np.isposinf(lower["threshold"])
    assert np.isneginf(upper["threshold"])


def test_lower_extension_fills_bins_above_synthetic_82_percent_cliff(monkeypatch):
    monkeypatch.setattr(coverage_search, "update_policy_params", lambda *args: None)
    calibration_scores = [0.0, 0.5, 1.0]
    # All thresholds in the calibration range yield 82% help. The other 18%
    # have test scores below the calibration minimum and become reachable only
    # when the finite threshold range is extended downward.
    episode_scores = np.concatenate([np.full(82, 1.5), np.linspace(-30.0, -1.0, 18)])
    policy = make_threshold_policy(calibration_scores=calibration_scores)
    baseline = make_sampler(policy, FakeEvaluator(episode_scores))
    extension_policy = make_threshold_policy(calibration_scores=calibration_scores)
    extended = make_sampler(
        extension_policy,
        FakeEvaluator(episode_scores),
        level_threshold_min=-30.0,
    )

    sample_percentiles = [0.25, 0.6, 0.9]
    baseline_afhps = [baseline.eval_at_percentile(p)[0] for p in sample_percentiles]
    extended_afhps = [extended.eval_at_percentile(p)[0] for p in sample_percentiles]
    extended_bins = {extended.determine_bin(afhp) for afhp in extended_afhps}
    baseline_bins = {baseline.determine_bin(afhp) for afhp in baseline_afhps}

    assert max(baseline_afhps) == pytest.approx(0.82)
    assert extended_bins.issuperset({17, 18, 19})
    assert max(baseline_bins) == 16

    _, _, extended_point = extended.eval_at_percentile(0.75)
    mapping = extended_point["level_threshold_mapping"]
    assert extended_point["threshold"] == pytest.approx(-22.25)
    assert mapping["calibration_min"] == 0.0
    assert mapping["calibration_max"] == 1.0
    assert mapping["requested_min"] == -30.0
    assert mapping["effective_min"] == -30.0
    assert mapping["effective_max"] == 1.0

    sampled_thresholds = [
        extended.eval_at_percentile(p)[2]["threshold"] for p in (0.25, 0.5, 0.75)
    ]
    assert sampled_thresholds[0] >= sampled_thresholds[1] >= sampled_thresholds[2]
    _, _, lower = extended.eval_at_lower_extreme()
    _, _, upper = extended.eval_at_upper_extreme()
    assert np.isposinf(lower["threshold"])
    assert np.isneginf(upper["threshold"])


def test_extended_binary_sampler_fills_all_bins_on_dense_scores(monkeypatch):
    monkeypatch.setattr(coverage_search, "update_policy_params", lambda *args: None)
    calibration_scores = np.linspace(0.0, 1.0, 101)
    episode_scores = np.concatenate(
        [
            np.linspace(0.0001, 1.0, 8200),
            np.linspace(-30.0, -0.0001, 1800),
        ]
    )
    evaluator = FakeEvaluator(episode_scores, max_calls=100)
    sampler = coverage_search.create_level_afhp_threshold_sampler(
        policy=make_threshold_policy(calibration_scores=calibration_scores),
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.05,
        level_threshold_min=-30.0,
    )

    result = sampler.run()
    filled_bins = {sampler.determine_bin(point.afhp) for point in result.points}

    assert filled_bins == set(range(20))
    assert any(point.afhp > 0.82 for point in result.points)
    assert evaluator.thresholds
    assert len(evaluator.thresholds) < 100


@pytest.mark.parametrize(
    "metric, calibration_scores, requested_min, message",
    [
        ("max_prob", [0.0, 0.5, 1.0], -30.0, "max_logit metric"),
        ("max_logit", None, -30.0, "calibrated episode maximum scores"),
        ("max_logit", [0.0, 0.5, 1.0], 0.0, "strictly below"),
        ("max_logit", [0.0, 0.5, 1.0], 1.0, "strictly below"),
        ("max_logit", [0.0, 0.5, 1.0], np.nan, "must be finite"),
    ],
)
def test_invalid_extension_requests_are_rejected(
    metric, calibration_scores, requested_min, message
):
    policy = make_threshold_policy(metric, calibration_scores)

    with pytest.raises(ValueError, match=message):
        make_sampler(
            policy,
            FakeEvaluator([0.0, 1.0]),
            level_threshold_min=requested_min,
        )


def test_extension_rejects_non_threshold_policies():
    with pytest.raises(ValueError, match="only for ThresholdPolicy"):
        coverage_search.create_level_afhp_threshold_sampler(
            policy=SimpleNamespace(),
            evaluator=FakeEvaluator([0.0, 1.0]),
            envs_factory=lambda: object(),
            split="test",
            level_threshold_min=-30.0,
        )


def test_eval_afhp_flag_parses_negative_threshold_min(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_afhp.py", "-level_threshold_min", "-30"],
    )

    args = flags.make()

    assert args.evaluation.level_threshold_min == -30.0

from types import SimpleNamespace

import numpy as np
import torch

from YRC.core.configs.global_configs import set_global_variable
from YRC.coverage.coverage_search import (
    ImageSVDDProbeSampler,
    ImageSVDDRawThresholdSampler,
    create_level_afhp_threshold_sampler,
    image_svdd_calibration_diagnostics,
    image_svdd_probe_decision,
)
from YRC.policies.ood import OODPolicy


class FakeAgent:
    def eval(self):
        return self


class FakePolicyEnv:
    num_envs = 1
    weak_agent = FakeAgent()


def make_ood_policy(feature_type="obs", clf_name="DeepSVDD", scores=None):
    set_global_variable("benchmark", "procgen")
    set_global_variable("device", torch.device("cpu"))
    config = SimpleNamespace(
        coord_policy=SimpleNamespace(
            collect_data_agent="weak",
            feature_type=feature_type,
            rolling_average="none",
            rolling_average_size=10,
        )
    )
    policy = OODPolicy(config, FakePolicyEnv())
    policy.clf_name = clf_name
    if scores is not None:
        policy._train_episode_max_scores = np.asarray(scores, dtype=float)
    return policy


def test_image_svdd_degenerate_calibration_is_detected():
    policy = make_ood_policy(scores=[0.32, 0.32, 0.32])

    diagnostics = image_svdd_calibration_diagnostics(policy)

    assert diagnostics["is_image_svdd"]
    assert diagnostics["is_degenerate"]
    assert diagnostics["unique_count"] == 1
    assert diagnostics["max_score"] == 0.32


def test_non_degenerate_or_non_image_calibration_diagnostics():
    image_policy = make_ood_policy(scores=[0.32, 0.33, 0.34])
    latent_policy = make_ood_policy(feature_type="hidden", scores=[0.32, 0.32, 0.32])

    assert not image_svdd_calibration_diagnostics(image_policy)["is_degenerate"]
    assert not image_svdd_calibration_diagnostics(latent_policy)["is_image_svdd"]
    assert not image_svdd_calibration_diagnostics(latent_policy)["is_degenerate"]


def test_create_level_sampler_selects_image_svdd_probe_sampler():
    policy = make_ood_policy(scores=[0.32, 0.32, 0.32])

    sampler = create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=object(),
        envs_factory=lambda: None,
        split="test",
        coverage_fraction=0.25,
    )

    assert isinstance(sampler, ImageSVDDProbeSampler)


def test_create_level_sampler_selects_probe_sampler_for_non_degenerate_image_svdd():
    policy = make_ood_policy(scores=[0.31, 0.32, 0.33, 0.34])

    sampler = create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=object(),
        envs_factory=lambda: None,
        split="test",
        coverage_fraction=0.25,
    )

    assert isinstance(sampler, ImageSVDDProbeSampler)


def test_image_svdd_probe_decision_triggers_on_non_degenerate_high_afhp_probes():
    diagnostics = image_svdd_calibration_diagnostics(
        make_ood_policy(scores=[0.32, 0.33, 0.34])
    )
    finite_probe_points = [
        {"threshold": 0.3201, "afhp": 1.0},
        {"threshold": 0.325, "afhp": 1.0},
        {"threshold": 0.33, "afhp": 1.0},
    ]

    decision = image_svdd_probe_decision(
        diagnostics,
        finite_probe_points,
        coverage_fraction=0.25,
    )

    assert decision["should_expand"]
    assert decision["reason"] == "all_finite_probes_high_afhp"


def test_image_svdd_probe_decision_keeps_healthy_percentile_search():
    diagnostics = image_svdd_calibration_diagnostics(
        make_ood_policy(scores=[0.32, 0.33, 0.34])
    )
    finite_probe_points = [
        {"threshold": 0.3201, "afhp": 0.95},
        {"threshold": 0.325, "afhp": 0.55},
        {"threshold": 0.33, "afhp": 0.10},
    ]

    decision = image_svdd_probe_decision(
        diagnostics,
        finite_probe_points,
        coverage_fraction=0.25,
    )

    assert not decision["should_expand"]
    assert decision["reason"] == "probe_percentile_search_healthy"


def test_raw_threshold_sampler_searches_above_id_threshold():
    evaluated_thresholds = []

    def eval_with_threshold(threshold):
        evaluated_thresholds.append(threshold)
        if threshold <= 0.32:
            afhp = 1.0
        else:
            afhp = min(max((0.42 - threshold) / 0.10, 0.0), 1.0)
        return afhp, 10.0 * afhp, {"threshold": threshold}

    sampler = ImageSVDDRawThresholdSampler(
        eval_with_threshold=eval_with_threshold,
        id_threshold=0.32,
        coverage_fraction=0.25,
        max_total_evals=20,
        expansion_initial_delta_fraction=0.1,
    )

    result = sampler.run()

    assert any(
        np.isfinite(threshold) and threshold > 0.32
        for threshold in evaluated_thresholds
    )
    assert result.info["threshold_strategy"] == "expand_above_id"
    assert result.info["bins_filled"] >= 3
    assert result.total_evals <= 20


def test_raw_threshold_sampler_picks_threshold_adjacent_pair_across_afhp_boundary():
    sampler = ImageSVDDRawThresholdSampler(
        eval_with_threshold=lambda threshold: (_ for _ in ()).throw(
            AssertionError("eval_with_threshold should not be called in this test")
        ),
        id_threshold=0.320000022649765,
        coverage_fraction=0.2,
        max_total_evals=30,
    )

    # Reconstruct a cluster smoke-test point cloud after 21 raw
    # evaluations. Order of insertion matters: stable-sort tie-breaking on
    # AFHP-only used to put eval 7 at the head of the AFHP=1 cluster, even
    # though eval 21 (added later) is the one threshold-adjacent to eval 20.
    eval_7 = {
        "threshold": 0.320000022649765,
        "afhp": 1.0,
        "performance": 1.0,
        "meta": {},
    }
    eval_8 = {
        "threshold": float(np.nextafter(0.320000022649765, float("inf"))),
        "afhp": 1.0,
        "performance": 1.0,
        "meta": {},
    }
    # Earlier expansion / bisects: AFHP=0 with larger thresholds than eval 20.
    earlier_afhp0 = [
        {"threshold": 0.32003202, "afhp": 0.0, "performance": 0.0, "meta": {}},
        {"threshold": 0.32000050, "afhp": 0.0, "performance": 0.0, "meta": {}},
    ]
    eval_20 = {
        "threshold": 0.3200000382747661,
        "afhp": 0.0,
        "performance": 0.0,
        "meta": {},
    }
    eval_21 = {
        "threshold": 0.32000003046226555,
        "afhp": 1.0,
        "performance": 1.0,
        "meta": {},
    }

    insertion_order = [eval_7, eval_8, *earlier_afhp0, eval_20, eval_21]
    for order, point in enumerate(insertion_order, start=1):
        point["key"] = sampler._threshold_key(point["threshold"])
        point["order"] = order
        sampler.points.append(point)
        sampler.seen_thresholds.add(point["key"])

    interval = sampler._select_widest_fillable_interval()

    assert interval is not None
    left, right = interval
    # Boundary pair must be the threshold-adjacent eval-20 / eval-21 pair.
    assert left["threshold"] == eval_20["threshold"]
    assert right["threshold"] == eval_21["threshold"]

    # Their midpoint must not collide with any previously-seen threshold,
    # otherwise _fill_afhp_bins would still bail with duplicate_threshold.
    midpoint = sampler._midpoint_threshold(left["threshold"], right["threshold"])
    assert midpoint is not None
    assert sampler._threshold_key(midpoint) not in sampler.seen_thresholds


def test_raw_threshold_sampler_stops_on_unfillable_step_function():
    def eval_with_threshold(threshold):
        afhp = 1.0 if threshold <= 0.32 else 0.0
        return afhp, 10.0 * afhp, {"threshold": threshold}

    sampler = ImageSVDDRawThresholdSampler(
        eval_with_threshold=eval_with_threshold,
        id_threshold=0.32,
        coverage_fraction=0.25,
        max_total_evals=20,
        expansion_initial_delta_fraction=0.1,
    )

    result = sampler.run()

    assert result.total_evals < 20
    assert result.early_stop_reason == "unfillable_afhp_intervals"
    assert result.info["unfillable_afhp_intervals"]

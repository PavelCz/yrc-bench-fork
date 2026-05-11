from types import SimpleNamespace

import numpy as np
import torch

from YRC.core.configs.global_configs import set_global_variable
from YRC.coverage.coverage_search import (
    ImageSVDDRawThresholdSampler,
    create_level_afhp_threshold_sampler,
    image_svdd_calibration_diagnostics,
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


def test_non_degenerate_or_non_image_calibration_does_not_trigger():
    image_policy = make_ood_policy(scores=[0.32, 0.33, 0.34])
    latent_policy = make_ood_policy(feature_type="hidden", scores=[0.32, 0.32, 0.32])

    assert not image_svdd_calibration_diagnostics(image_policy)["is_degenerate"]
    assert not image_svdd_calibration_diagnostics(latent_policy)["is_image_svdd"]
    assert not image_svdd_calibration_diagnostics(latent_policy)["is_degenerate"]


def test_create_level_sampler_selects_image_svdd_raw_threshold_fallback():
    policy = make_ood_policy(scores=[0.32, 0.32, 0.32])

    sampler = create_level_afhp_threshold_sampler(
        policy=policy,
        evaluator=object(),
        envs_factory=lambda: None,
        split="test",
        coverage_fraction=0.25,
    )

    assert isinstance(sampler, ImageSVDDRawThresholdSampler)


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

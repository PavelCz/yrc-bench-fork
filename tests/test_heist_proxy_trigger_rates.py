from pathlib import Path

import numpy as np
import pytest

from analyzing.heist_proxy_trigger_rates import (
    EndpointTriggerRates,
    calculate_rates,
    discover_artifacts,
    experiment_means,
    load_endpoint_rates,
    median_iqr,
)


def endpoint_summary(keys_collected):
    return {
        "keys_collected": keys_collected,
        "num_keys": [2, 6, 6],
        "total_chests": [4, 3, 3],
        "level_ood_gt": [False, True, True],
    }


def write_artifact(path: Path, afhps=(0.0, 0.5, 1.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = [
        endpoint_summary([2, 4, 5]),
        endpoint_summary([2, 3, 5]),
        endpoint_summary([2, 3, 6]),
    ]
    meta = np.asarray(
        [{"summary": {"test": summary}} for summary in summaries[: len(afhps)]],
        dtype=object,
    )
    np.savez(path, afhps=np.asarray(afhps), meta=meta)


def test_load_endpoint_rates_uses_ood_episodes_only(tmp_path):
    artifact = tmp_path / "eval_seed_1_test.npz"
    write_artifact(artifact)

    novice, expert = load_endpoint_rates(artifact, exp_id=2, method="max-prob")

    assert novice.role == "novice"
    assert novice.ood_episodes == 2
    assert novice.redundant_key_rate == 1.0
    assert novice.all_keys_rate == 0.0
    assert expert.role == "expert"
    assert expert.redundant_key_rate == 0.5
    assert expert.all_keys_rate == 0.5


def test_discovery_selects_latest_matching_campaign_run(tmp_path):
    method_dir = tmp_path / "campaign_heist_exp0" / "heist_max-prob_exp0"
    old_artifact = method_dir / "20260101_000000" / "old_test.npz"
    latest_artifact = method_dir / "20260102_000000" / "latest_test.npz"
    write_artifact(old_artifact)
    write_artifact(latest_artifact)
    write_artifact(
        tmp_path
        / "other_heist_exp0"
        / "heist_max-prob_exp0"
        / "20260103_000000"
        / "other_test.npz"
    )

    artifacts = discover_artifacts(tmp_path, prefixes=["campaign"])

    assert artifacts == {(0, "max-prob"): latest_artifact}
    assert len(calculate_rates(artifacts)) == 2


def test_load_endpoint_rates_requires_both_extremes(tmp_path):
    artifact = tmp_path / "eval_seed_1_test.npz"
    write_artifact(artifact, afhps=(0.0, 0.5))

    with pytest.raises(ValueError, match="no AFHP=1 expert endpoint"):
        load_endpoint_rates(artifact, exp_id=0, method="max-prob")


def test_experiment_aggregation_averages_methods_before_median():
    rows = [
        EndpointTriggerRates(0, "a", "novice", 10, 0.8, 0.6, Path("a")),
        EndpointTriggerRates(0, "b", "novice", 10, 1.0, 0.8, Path("b")),
        EndpointTriggerRates(1, "a", "novice", 10, 0.7, 0.5, Path("c")),
    ]

    means = experiment_means(rows)

    assert means[(0, "novice")] == pytest.approx((2, 0.9, 0.7))
    assert means[(1, "novice")] == pytest.approx((1, 0.7, 0.5))
    assert median_iqr([means[(0, "novice")][1], means[(1, "novice")][1]]) == (
        pytest.approx(0.8),
        pytest.approx(0.75),
        pytest.approx(0.85),
    )

from pathlib import Path

import numpy as np
import pytest

from YRC.core import eval_script_utils
from YRC.coverage import coverage_search


class FakePolicy:
    def train_percentile_step(self, percentile):
        return (100.0 - percentile) / 100.0

    def train_percentile_level(self, percentile):
        return (100.0 - percentile) / 100.0


class FakeEvaluator:
    def __init__(self, *, fail_on_call, level_afhp=1.0):
        self.fail_on_call = fail_on_call
        self.level_afhp = level_afhp
        self.calls = 0

    def eval(self, policy, envs, splits, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated evaluation failure")

        split = splits[0]
        is_ood = self.level_afhp > 0.0
        return {
            split: {
                "level_ood_pred": [is_ood],
                "action_1_frac": 0.25,
                "env_return_mean": float(self.calls),
                "heist_outcomes": np.asarray([self.calls]),
            }
        }


def _assert_partial_artifact(path, *, expected_points):
    with np.load(path, allow_pickle=True) as saved:
        assert set(saved.files) == {
            "afhps",
            "performances",
            "desired_percentiles",
            "meta",
            "order",
            "sampling_info",
        }
        assert len(saved["afhps"]) == expected_points
        assert len(saved["performances"]) == expected_points
        assert len(saved["desired_percentiles"]) == expected_points
        assert len(saved["meta"]) == expected_points
        assert len(saved["order"]) == expected_points
        assert saved["sampling_info"].item()["status"] == "in_progress"
        assert saved["sampling_info"].item()["completed_evaluations"] == expected_points
        assert saved["meta"][0]["summary"]["test"]["heist_outcomes"].tolist()


def test_step_sampler_checkpoints_completed_evaluation_before_next_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(coverage_search, "update_policy_params", lambda *args: None)
    output_path = tmp_path / "step_test.npz"
    checkpoints = eval_script_utils.IntermediateEvaluationResults(output_path)
    evaluator = FakeEvaluator(fail_on_call=2, level_afhp=0.0)
    sampler = coverage_search.create_step_afhp_threshold_sampler(
        policy=FakePolicy(),
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.25,
        on_evaluation=checkpoints.record_evaluation,
    )

    with pytest.raises(RuntimeError, match="simulated evaluation failure"):
        sampler.run()

    _assert_partial_artifact(output_path, expected_points=1)


def test_raw_image_svdd_sampler_checkpoints_direct_threshold_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(coverage_search, "update_policy_params", lambda *args: None)
    monkeypatch.setattr(
        coverage_search,
        "image_svdd_calibration_diagnostics",
        lambda policy: {
            "is_image_svdd": True,
            "is_degenerate": True,
            "max_score": 0.32,
            "unique_count": 1,
            "tolerance": 1e-8,
        },
    )
    output_path = tmp_path / "level_test.npz"
    checkpoints = eval_script_utils.IntermediateEvaluationResults(output_path)
    evaluator = FakeEvaluator(fail_on_call=8)
    sampler = coverage_search.create_level_afhp_threshold_sampler(
        policy=FakePolicy(),
        evaluator=evaluator,
        envs_factory=lambda: object(),
        split="test",
        coverage_fraction=0.25,
        on_evaluation=checkpoints.record_evaluation,
    )

    with pytest.raises(RuntimeError, match="simulated evaluation failure"):
        sampler.run()

    _assert_partial_artifact(output_path, expected_points=7)
    with np.load(output_path, allow_pickle=True) as saved:
        assert np.isnan(saved["desired_percentiles"][-1])
        assert saved["meta"][-1]["threshold"] == 0.32


def test_atomic_npz_save_keeps_previous_checkpoint_when_write_fails(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "results.npz"
    eval_script_utils.save_npz_results(output_path, values=np.asarray([1, 2]))
    previous_bytes = output_path.read_bytes()

    def fail_after_partial_write(file_obj, **arrays):
        file_obj.write(b"partial archive")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(eval_script_utils.np, "savez", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated interrupted write"):
        eval_script_utils.save_npz_results(output_path, values=np.asarray([3, 4]))

    assert output_path.read_bytes() == previous_bytes
    assert not list(tmp_path.glob(".results.npz.*.tmp"))
    with np.load(output_path) as saved:
        assert saved["values"].tolist() == [1, 2]


def test_save_npz_results_preserves_numpy_extension_behavior(tmp_path):
    output_path = Path(tmp_path / "results")

    eval_script_utils.save_npz_results(output_path, values=np.asarray([1]))

    with np.load(tmp_path / "results.npz") as saved:
        assert saved["values"].tolist() == [1]

"""Tests for the calibration auto-insertion logic in scripts/run_eval.py."""

import contextlib
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_eval  # noqa: E402


def _call(
    method: str,
    *,
    calibration_exists: bool,
    svdd_model_path,
    submit_return: str = "7777",
    dry_run: bool = False,
):
    checkpoints = {
        "sim": "/tmp/weak.pt",
        "weak": "/tmp/weak.pt",
        "strong": "/tmp/strong.pt",
    }
    calibration_path = Path("/tmp/coord/maze/exp0/method/runkey/calibration.npz")

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                Path,
                "exists",
                autospec=True,
                side_effect=lambda self: (
                    calibration_exists if self == calibration_path else True
                ),
            )
        )
        submit_mock = stack.enter_context(
            mock.patch("scripts.run_eval._sbatch_submit", return_value=submit_return)
        )
        result = run_eval._maybe_submit_calibration_job(
            method=method,
            calibration_path=calibration_path,
            coordination_artifact_dir=calibration_path.parent,
            experiment_group="test_maze_exp0",
            config_path="configs/eval/maze/max_prob.yaml",
            checkpoints=checkpoints,
            level_seeds_file=Path("/tmp/seeds/0.json"),
            cp_feature=None,
            svdd_model_path=svdd_model_path,
            ensemble_members=None,
            conda_env="ood-stable",
            log_dir=calibration_path.parent / "slurm",
            qos="default",
            env="maze",
            exp_id=0,
            dry_run=dry_run,
        )
    return result, submit_mock


class TestCacheHitNoPrep(unittest.TestCase):
    def test_non_svdd_cache_hit_returns_none(self):
        result, submit_mock = _call(
            "max-prob", calibration_exists=True, svdd_model_path=None
        )
        self.assertIsNone(result)
        submit_mock.assert_not_called()

    def test_svdd_cache_hit_returns_none(self):
        result, submit_mock = _call(
            "svdd-latent",
            calibration_exists=True,
            svdd_model_path="/tmp/svdd/trained.joblib",
        )
        self.assertIsNone(result)
        submit_mock.assert_not_called()


class TestCacheMissNonSvdd(unittest.TestCase):
    def test_cache_miss_submits_calibration_and_returns_job_id(self):
        result, submit_mock = _call(
            "max-prob",
            calibration_exists=False,
            svdd_model_path=None,
            submit_return="9999",
        )
        self.assertEqual(result, "9999")
        self.assertEqual(submit_mock.call_count, 1)
        submitted_script = submit_mock.call_args.args[0]
        self.assertIn("calibrate_afhp.py", submitted_script)

    def test_dry_run_returns_sentinel_and_does_not_submit(self):
        result, submit_mock = _call(
            "max-prob",
            calibration_exists=False,
            svdd_model_path=None,
            dry_run=True,
        )
        self.assertEqual(result, run_eval._DRY_RUN_CALIB_JOB_ID)
        submit_mock.assert_not_called()


class TestCacheMissSvdd(unittest.TestCase):
    def test_svdd_cache_miss_with_trained_model_submits_calibration(self):
        result, submit_mock = _call(
            "svdd-latent",
            calibration_exists=False,
            svdd_model_path="/tmp/svdd/trained.joblib",
            submit_return="8888",
        )
        self.assertEqual(result, "8888")
        submitted_script = submit_mock.call_args.args[0]
        self.assertIn("calibrate_afhp.py", submitted_script)
        # SVDD passes just the basename via -f_n; the worker resolves it
        # against experiment_dir at runtime.
        self.assertIn("-f_n trained.joblib", submitted_script)

    def test_svdd_cache_miss_without_trained_model_returns_skip_sentinel(self):
        # When svdd_model_path is None, the method short-circuits and submits nothing.
        with mock.patch("scripts.run_eval._sbatch_submit") as submit_mock:
            result = run_eval._maybe_submit_calibration_job(
                method="svdd-latent",
                calibration_path=Path("/tmp/nonexistent/calibration.npz"),
                coordination_artifact_dir=Path("/tmp/nonexistent"),
                experiment_group="test_maze_exp0",
                config_path="configs/eval/maze/latent_svdd.yaml",
                checkpoints={
                    "sim": "/tmp/weak.pt",
                    "weak": "/tmp/weak.pt",
                    "strong": "/tmp/strong.pt",
                },
                level_seeds_file=Path("/tmp/seeds/0.json"),
                cp_feature="hidden",
                svdd_model_path=None,
                ensemble_members=None,
                conda_env="ood-stable",
                log_dir=Path("/tmp/slurm"),
                qos="default",
                env="maze",
                exp_id=0,
                dry_run=False,
            )
        self.assertIs(result, run_eval._SKIP_EXP)
        submit_mock.assert_not_called()


class TestEvalSbatchDependencyLine(unittest.TestCase):
    """Confirm the eval sbatch builders emit --dependency=afterok when given a job id."""

    def _base_eval_args(self):
        return {
            "config": "configs/eval/maze/max_prob.yaml",
            "name": "maze_max_prob_exp0",
            "experiment_group": "test_maze_exp0",
            "num_levels": 50,
            "video_episodes_to_collect": 4,
            "video_filter": "all",
            "cp_rolling_average": "none",
            "video_logging_mode": "folder",
            "video_filter_mode": "any",
            "num_bins": 20,
            "wandb_project": None,
            "level_seeds_file": "/tmp/seeds/0.json",
            "svdd_model_path": None,
            "cp_feature": None,
            "ensemble_members": None,
            "sim": "/tmp/weak.pt",
            "weak": "/tmp/weak.pt",
            "strong": "/tmp/strong.pt",
        }

    def test_sequential_without_dependency(self):
        script = run_eval.build_sequential_eval_sbatch_command(
            "maze_max_prob_exp0",
            self._base_eval_args(),
            "ood-stable",
            Path("/tmp/log"),
            Path("/tmp/cal.npz"),
        )
        self.assertNotIn("--dependency=afterok", script)

    def test_sequential_with_dependency(self):
        script = run_eval.build_sequential_eval_sbatch_command(
            "maze_max_prob_exp0",
            self._base_eval_args(),
            "ood-stable",
            Path("/tmp/log"),
            Path("/tmp/cal.npz"),
            dependency_job_id="1234",
        )
        self.assertIn("#SBATCH --dependency=afterok:1234", script)

    def test_bin_array_without_dependency(self):
        script = run_eval.build_bin_array_sbatch_command(
            "maze_max_prob_exp0",
            self._base_eval_args(),
            "ood-stable",
            Path("/tmp/log"),
            Path("/tmp/cal.npz"),
            num_bins=20,
        )
        self.assertNotIn("--dependency=afterok", script)

    def test_bin_array_with_dependency(self):
        script = run_eval.build_bin_array_sbatch_command(
            "maze_max_prob_exp0",
            self._base_eval_args(),
            "ood-stable",
            Path("/tmp/log"),
            Path("/tmp/cal.npz"),
            num_bins=20,
            dependency_job_id="5678",
        )
        self.assertIn("#SBATCH --dependency=afterok:5678", script)


if __name__ == "__main__":
    unittest.main()

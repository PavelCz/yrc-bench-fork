import contextlib
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_svdd  # noqa: E402


class TestTrainSvddDryRun(unittest.TestCase):
    """Dry-run tests: confirm the gather -> train chain is constructed."""

    def _run_dry_run(self, method: str):
        checkpoints = {
            "sim": "/tmp/weak.pt",
            "weak": "/tmp/weak.pt",
            "strong": "/tmp/strong.pt",
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "scripts.train_svdd.get_checkpoints",
                    return_value=dict(checkpoints),
                )
            )
            stack.enter_context(
                mock.patch("scripts.train_svdd.Path.exists", return_value=True)
            )
            submit_mock = stack.enter_context(
                mock.patch("scripts.train_svdd.sbatch_submit")
            )
            ok = train_svdd.submit_svdd_prep_for_exp(
                env="maze",
                exp_id=0,
                method=method,
                prefix="test",
                checkpoint_base_path="/tmp/ckpt",
                seeds_base_path="/tmp/seeds",
                coordination_root=Path("/tmp/coord"),
                conda_env="ood-stable",
                qos="default",
                num_rollouts=4,
                query_cost=0.0,
                gather_config="configs/procgen_gather.yaml",
                train_config="configs/procgen_ood.yaml",
                checkpoint_overrides={"sim": None, "weak": None, "strong": None},
                dry_run=True,
            )
        submit_mock.assert_not_called()
        self.assertTrue(ok)

    def test_svdd_image_dry_run(self):
        self._run_dry_run("svdd-image")

    def test_svdd_latent_dry_run(self):
        self._run_dry_run("svdd-latent")


class TestTrainSvddSubmitChain(unittest.TestCase):
    """Confirm gather is submitted first and train depends on it via afterok."""

    def test_train_depends_on_gather_job_id(self):
        captured_scripts = []

        def fake_submit(script: str, log_dir: Path):
            captured_scripts.append(script)
            if "gather" in script and "train" not in script:
                return "1001"
            return "1002"

        checkpoints = {
            "sim": "/tmp/weak.pt",
            "weak": "/tmp/weak.pt",
            "strong": "/tmp/strong.pt",
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "scripts.train_svdd.get_checkpoints",
                    return_value=dict(checkpoints),
                )
            )
            stack.enter_context(
                mock.patch("scripts.train_svdd.Path.exists", return_value=True)
            )
            stack.enter_context(
                mock.patch("scripts.train_svdd.sbatch_submit", side_effect=fake_submit)
            )
            ok = train_svdd.submit_svdd_prep_for_exp(
                env="maze",
                exp_id=0,
                method="svdd-latent",
                prefix="test",
                checkpoint_base_path="/tmp/ckpt",
                seeds_base_path="/tmp/seeds",
                coordination_root=Path("/tmp/coord"),
                conda_env="ood-stable",
                qos="default",
                num_rollouts=4,
                query_cost=0.0,
                gather_config="configs/procgen_gather.yaml",
                train_config="configs/procgen_ood.yaml",
                checkpoint_overrides={"sim": None, "weak": None, "strong": None},
                dry_run=False,
            )

        self.assertTrue(ok)
        self.assertEqual(len(captured_scripts), 2)
        gather_script, train_script = captured_scripts
        self.assertIn("gather", gather_script)
        self.assertNotIn("afterok", gather_script)
        self.assertIn("train", train_script)
        self.assertIn("--dependency=afterok:1001", train_script)


class TestTrainSvddRejectsNonSvdd(unittest.TestCase):
    """argparse choices restrict --method to SVDD; verify via main()."""

    def test_main_rejects_non_svdd_method_via_choices(self):
        argv = [
            "train_svdd.py",
            "--env",
            "maze",
            "--method",
            "wait",
            "--prefix",
            "test",
            "--exp-ids",
            "0",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                train_svdd.main()
        # argparse exits with code 2 on invalid choice.
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

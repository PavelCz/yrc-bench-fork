from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from YRC.core.artifacts import (  # noqa: E402
    resolve_metadata_path,
    write_coordination_metadata,
)
from scripts.run_prepare_coordination import (  # noqa: E402
    PreparePlan,
    build_coordination_metadata,
    get_prepare_plan,
)


class TestRunPrepareCoordination(unittest.TestCase):
    def test_svdd_methods_require_rollouts_and_training(self):
        for method in ("svdd-image", "svdd-latent"):
            plan = get_prepare_plan(method)
            self.assertTrue(plan.requires_rollouts)
            self.assertTrue(plan.requires_training)
            self.assertTrue(plan.requires_calibration)

    def test_non_trainable_eval_methods_only_require_calibration(self):
        for method in (
            "max-prob",
            "max-logit",
            "lb-random",
            "ts-random",
            "ensemble",
            "ensemble-single",
            "wait",
        ):
            plan = get_prepare_plan(method)
            self.assertFalse(plan.requires_rollouts)
            self.assertFalse(plan.requires_training)
            self.assertTrue(plan.requires_calibration)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            get_prepare_plan("ppo")

    def test_build_coordination_metadata_records_identity_and_phases(self):
        metadata = build_coordination_metadata(
            env="maze",
            exp_id=3,
            method="svdd-image",
            method_name="svdd_image",
            experiment_group="prep_maze_exp3",
            coordination_artifact_dir=Path("/tmp/coord/maze/exp3/svdd_image/run0"),
            calibration_path=Path(
                "/tmp/coord/maze/exp3/svdd_image/run0/calibration.npz"
            ),
            plan=PreparePlan(requires_rollouts=True, requires_training=True),
            checkpoints={"sim": "/sim", "weak": "/weak", "strong": "/strong"},
            level_seeds_file=Path("/seeds/3.json"),
            feature_type="image",
            ensemble_members=None,
        )

        self.assertEqual(metadata["method"], "svdd-image")
        self.assertEqual(metadata["run_key"], "prep_maze_exp3")
        self.assertEqual(metadata["phases"], ["gather_rollouts", "train", "calibrate"])
        self.assertEqual(metadata["phase_status"]["calibrate"], "pending")
        self.assertEqual(metadata["acting_policies"]["weak"], "/weak")

    def test_write_coordination_metadata_uses_canonical_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coordination_dir = Path(tmpdir) / "coordination-run"
            metadata = {"method": "wait", "phase_status": {"calibrate": "pending"}}

            written_path = write_coordination_metadata(coordination_dir, metadata)

            self.assertEqual(written_path, resolve_metadata_path(coordination_dir))
            self.assertTrue(written_path.exists())


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from YRC.core.artifacts import (  # noqa: E402
    resolve_calibration_path,
    resolve_coordination_artifact_dir,
)


class TestCalibrationPathNaming(unittest.TestCase):
    def setUp(self) -> None:
        self.coordination_root = Path("/tmp/coordination-policies")
        self.env = "maze"
        self.exp_id = 0
        self.method_name = "max_prob"
        self.experiment_group = "run0"

    def test_uses_coordination_artifact_dir(self):
        coordination_dir = resolve_coordination_artifact_dir(
            self.env,
            self.exp_id,
            self.method_name,
            self.experiment_group,
            coordination_root=self.coordination_root,
        )
        path = resolve_calibration_path(coordination_dir)

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp0"
            / self.method_name
            / self.experiment_group
            / "calibration.npz",
        )

    def test_keeps_wait_method_in_its_own_artifact_dir(self):
        coordination_dir = resolve_coordination_artifact_dir(
            self.env,
            3,
            "wait",
            self.experiment_group,
            coordination_root=self.coordination_root,
        )
        path = resolve_calibration_path(coordination_dir)

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp3"
            / "wait"
            / self.experiment_group
            / "calibration.npz",
        )

    def test_keeps_svdd_method_in_its_own_artifact_dir(self):
        coordination_dir = resolve_coordination_artifact_dir(
            self.env,
            2,
            "svdd_latent",
            self.experiment_group,
            coordination_root=self.coordination_root,
        )
        path = resolve_calibration_path(coordination_dir)

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp2"
            / "svdd_latent"
            / self.experiment_group
            / "calibration.npz",
        )

    def test_run_key_is_part_of_artifact_identity(self):
        coordination_dir = resolve_coordination_artifact_dir(
            self.env,
            self.exp_id,
            self.method_name,
            "run4",
            coordination_root=self.coordination_root,
        )
        path = resolve_calibration_path(coordination_dir)

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp0"
            / self.method_name
            / "run4"
            / "calibration.npz",
        )


if __name__ == "__main__":
    unittest.main()

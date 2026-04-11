from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval import resolve_calibration_path  # noqa: E402


class TestCalibrationPathNaming(unittest.TestCase):
    def setUp(self) -> None:
        self.coordination_root = Path("/tmp/coordination-policies")
        self.env = "maze"
        self.exp_id = 0
        self.method_name = "max_prob"
        self.experiment_group = "run0"

    def test_uses_coordination_artifact_dir_and_timestep_suffix(self):
        eval_args = {
            "weak": "/data/acting/model_200015872.pth",
            "sim": "/data/acting/model_100000.pth",
            "strong": "/data/acting/model_300000.pth",
            "svdd_model_path": None,
        }

        path = resolve_calibration_path(
            self.env,
            self.exp_id,
            self.method_name,
            self.experiment_group,
            "maze_max_prob_exp0",
            eval_args,
            coordination_root=self.coordination_root,
        )

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp0"
            / self.method_name
            / self.experiment_group
            / "maze_max_prob_exp0_calibration_200M.npz",
        )

    def test_no_timestep_falls_back_to_plain_name(self):
        eval_args = {
            "weak": "/data/acting/weak_latest.pth",
            "sim": "/data/acting/sim_latest.pth",
            "strong": "/data/acting/strong_latest.pth",
            "svdd_model_path": None,
        }

        path = resolve_calibration_path(
            self.env,
            3,
            "wait",
            self.experiment_group,
            "maze_wait_exp3",
            eval_args,
            coordination_root=self.coordination_root,
        )

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp3"
            / "wait"
            / self.experiment_group
            / "maze_wait_exp3_calibration.npz",
        )

    def test_prefers_svdd_timestep_when_available(self):
        eval_args = {
            "weak": "/data/acting/model_100000.pth",
            "sim": "/data/acting/model_110000.pth",
            "strong": "/data/acting/model_120000.pth",
            "svdd_model_path": "/data/acting/model_900000.pth",
        }

        path = resolve_calibration_path(
            self.env,
            2,
            "svdd_latent",
            self.experiment_group,
            "maze_svdd_latent_exp2",
            eval_args,
            coordination_root=self.coordination_root,
        )

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp2"
            / "svdd_latent"
            / self.experiment_group
            / "maze_svdd_latent_exp2_calibration_900k.npz",
        )

    def test_formats_exact_tens_of_millions_correctly(self):
        eval_args = {
            "weak": "/data/acting/model_10000000.pth",
            "sim": "/data/acting/model_1000.pth",
            "strong": "/data/acting/model_1000.pth",
            "svdd_model_path": None,
        }

        path = resolve_calibration_path(
            self.env,
            9,
            self.method_name,
            self.experiment_group,
            "maze_max_prob_exp9",
            eval_args,
            coordination_root=self.coordination_root,
        )

        self.assertEqual(
            path,
            self.coordination_root
            / self.env
            / "exp9"
            / self.method_name
            / self.experiment_group
            / "maze_max_prob_exp9_calibration_10M.npz",
        )


if __name__ == "__main__":
    unittest.main()

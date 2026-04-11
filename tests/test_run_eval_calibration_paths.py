from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval import resolve_calibration_path


class TestCalibrationPathNaming(unittest.TestCase):
    def test_uses_checkpoint_parent_and_timestep_suffix(self):
        eval_args = {
            "weak": "/tmp/checkpoints/model_200015872.pth",
            "sim": "/tmp/checkpoints/model_200015872.pth",
            "strong": "/tmp/checkpoints/model_200015872.pth",
            "svdd_model_path": None,
        }

        path = resolve_calibration_path("maze_max_prob_exp0", eval_args)

        self.assertEqual(
            path,
            Path("/tmp/checkpoints/maze_max_prob_exp0_calibration_200M.npz"),
        )

    def test_no_timestep_falls_back_to_plain_name(self):
        eval_args = {
            "weak": "/tmp/checkpoints/weak_latest.pth",
            "sim": "/tmp/checkpoints/sim_latest.pth",
            "strong": "/tmp/checkpoints/strong_latest.pth",
            "svdd_model_path": None,
        }

        path = resolve_calibration_path("maze_wait_exp3", eval_args)

        self.assertEqual(
            path,
            Path("/tmp/checkpoints/maze_wait_exp3_calibration.npz"),
        )


if __name__ == "__main__":
    unittest.main()

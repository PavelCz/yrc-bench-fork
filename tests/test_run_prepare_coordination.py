from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_prepare_coordination import get_prepare_plan  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

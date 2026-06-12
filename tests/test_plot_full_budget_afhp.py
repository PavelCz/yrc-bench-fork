import numpy as np

from analyzing.plot_full_budget_afhp import load_full_budget_data


def test_full_budget_help_only_is_restricted_to_original_level_seeds(tmp_path):
    original_npz = tmp_path / "eval_seed_1_test.npz"
    full_budget_npz = tmp_path / "eval_seed_1_test_full_budget_eval.npz"

    original_meta = np.array(
        [
            {
                "summary": {
                    "test": {
                        "level_seeds": [10, 11],
                        "level_ood_pred": [True, False],
                        "raw_returns": [1.0, 2.0],
                    }
                }
            },
            {
                "summary": {
                    "test": {
                        "level_seeds": [10, 11],
                        "level_ood_pred": [False, True],
                        "raw_returns": [3.0, 4.0],
                    }
                }
            },
        ],
        dtype=object,
    )
    np.savez(
        original_npz,
        afhps=np.array([50.0, 50.0]),
        performances=np.array([1.5, 3.5]),
        meta=original_meta,
    )

    full_budget_meta = np.array(
        [
            {
                "summary": {
                    "level_seeds": [10, 11, 12],
                    "level_ood_pred": [True, False, True],
                    "raw_returns": [10.0, 20.0, 120.0],
                }
            },
            {
                "summary": {
                    "level_seeds": [10, 11, 12],
                    "level_ood_pred": [False, True, True],
                    "raw_returns": [30.0, 40.0, 140.0],
                }
            },
        ],
        dtype=object,
    )
    np.savez(
        full_budget_npz,
        thresholds=np.array([0.1, 0.2]),
        original_afhps=np.array([50.0, 50.0]),
        full_budget_afhps=np.array([67.0, 67.0]),
        original_performances=np.array([1.5, 3.5]),
        full_budget_performances=np.array([50.0, 70.0]),
        full_budget_meta=full_budget_meta,
    )

    data = load_full_budget_data(full_budget_npz, original_npz)

    np.testing.assert_allclose(data["original_help_performances"], [1.0, 4.0])
    np.testing.assert_allclose(data["full_budget_help_performances"], [10.0, 40.0])

import importlib

import matplotlib
import numpy as np
import pytest


def ignore_backend_selection(*args, **kwargs):
    del args, kwargs


matplotlib.use = ignore_backend_selection
paper_plot = importlib.import_module("analyzing.paper_plot")
analysis_utils = importlib.import_module("analyzing.utils")
DATA_KEY_NAMES = paper_plot.DATA_KEY_NAMES
apply_level_split_to_y_key = paper_plot.apply_level_split_to_y_key
extract_from_data = analysis_utils.extract_from_data


def make_split_curve_data():
    first_test = {
        "raw_returns": [4.0, 2.0, 1.0],
        "level_ood_gt": [False, False, True],
        "level_ood_pred": [False, True, True],
    }
    second_test = {
        "raw_returns": [3.0, 0.0],
        "level_ood_gt": [True, True],
        "level_ood_pred": [True, False],
    }
    data = {
        "meta": np.array(
            [
                {"summary": {"test": first_test}},
                {"summary": {"test": second_test}},
            ],
            dtype=object,
        ),
        "afhps": np.array([0.0, 1.0]),
        "performances": np.array([np.mean([4.0, 2.0, 1.0]), np.mean([3.0, 0.0])]),
        "order": np.array([1, 2]),
    }
    return data


def test_id_and_ood_performance_use_ground_truth_split():
    data = make_split_curve_data()

    np.testing.assert_allclose(extract_from_data(data, "id_performance"), [3.0, np.nan])
    np.testing.assert_allclose(extract_from_data(data, "ood_performance"), [1.0, 1.5])


def test_id_and_ood_level_afhp_use_ground_truth_split():
    data = make_split_curve_data()

    np.testing.assert_allclose(extract_from_data(data, "id_level_afhp"), [0.5, np.nan])
    np.testing.assert_allclose(extract_from_data(data, "ood_level_afhp"), [1.0, 0.5])


def test_level_split_y_key_mapping():
    assert apply_level_split_to_y_key("performance", "id") == "id_performance"
    assert apply_level_split_to_y_key("performance", "ood") == "ood_performance"
    assert (
        apply_level_split_to_y_key("mean_oracle_regret", "ood")
        == "ood_mean_oracle_regret"
    )
    assert (
        apply_level_split_to_y_key("id_mean_oracle_regret", "ood")
        == "ood_mean_oracle_regret"
    )


def test_level_split_rejects_unsupported_y_key():
    with pytest.raises(ValueError, match="ood_accuracy"):
        apply_level_split_to_y_key("ood_accuracy", "id")


def test_level_split_plotting_labels_are_readable():
    assert DATA_KEY_NAMES["id_performance"] == "Mean Return (ID)"
    assert DATA_KEY_NAMES["ood_performance"] == "Mean Return (OOD)"
    assert DATA_KEY_NAMES["id_level_afhp"] == "Ask-For-Help Percentage (AFHP, ID)"
    assert DATA_KEY_NAMES["ood_level_afhp"] == "Ask-For-Help Percentage (AFHP, OOD)"

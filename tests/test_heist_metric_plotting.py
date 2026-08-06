import importlib

import matplotlib
import numpy as np
import pytest


# The analysis modules select TkAgg for interactive use. Keep this unit test
# headless without changing the production plotting backend.
def ignore_backend_selection(*args, **kwargs):
    del args, kwargs


matplotlib.use = ignore_backend_selection
paper_plot = importlib.import_module("analyzing.paper_plot")
analysis_utils = importlib.import_module("analyzing.utils")
DATA_KEY_NAMES = paper_plot.DATA_KEY_NAMES
HEIST_OUTCOME_DATA_KEYS = analysis_utils.HEIST_OUTCOME_DATA_KEYS
extract_from_data = analysis_utils.extract_from_data
extract_x_and_y_values = analysis_utils.extract_x_and_y_values


EXPECTED_LABELS = {
    "mean_oracle_regret": "Mean Oracle Regret",
    "id_mean_oracle_regret": "Mean Oracle Regret (ID)",
    "ood_mean_oracle_regret": "Mean Oracle Regret (OOD)",
    "mean_surplus_keys": "Mean Surplus Keys",
    "id_mean_surplus_keys": "Mean Surplus Keys (ID)",
    "ood_mean_surplus_keys": "Mean Surplus Keys (OOD)",
    "timeout_fraction": "Timeout Fraction",
    "id_timeout_fraction": "Timeout Fraction (ID)",
    "ood_timeout_fraction": "Timeout Fraction (OOD)",
}


def make_curve_data():
    first = {
        "mean_oracle_regret": 0.20,
        "id_mean_oracle_regret": 0.10,
        "ood_mean_oracle_regret": 0.30,
        "mean_surplus_keys": 1.50,
        "id_mean_surplus_keys": 0.25,
        "ood_mean_surplus_keys": 2.75,
        "timeout_fraction": 0.40,
        "id_timeout_fraction": 0.10,
        "ood_timeout_fraction": 0.70,
    }
    second = {key: value + 0.05 for key, value in first.items()}
    return (
        {
            "meta": np.array(
                [
                    {
                        "summary": {
                            "test": {
                                **first,
                                "level_ood_gt": [False, True],
                                "level_ood_pred": [False, False],
                            }
                        }
                    },
                    {
                        "summary": {
                            "test": {
                                **second,
                                "level_ood_gt": [False, True],
                                "level_ood_pred": [True, True],
                            }
                        }
                    },
                ],
                dtype=object,
            ),
            "afhps": np.array([0.0, 1.0]),
            "performances": np.array([1.0, 2.0]),
            "order": np.array([1, 2]),
        },
        first,
        second,
    )


def test_extracts_all_heist_outcome_plotting_keys():
    data, first, second = make_curve_data()

    assert HEIST_OUTCOME_DATA_KEYS == set(EXPECTED_LABELS)
    for key in HEIST_OUTCOME_DATA_KEYS:
        np.testing.assert_allclose(
            extract_from_data(data, key), [first[key], second[key]]
        )


def test_heist_outcome_plotting_labels_are_readable():
    assert {key: DATA_KEY_NAMES[key] for key in EXPECTED_LABELS} == EXPECTED_LABELS


def test_empty_heist_split_is_exposed_as_nan():
    data, _, _ = make_curve_data()
    data["meta"][0]["summary"]["test"]["ood_timeout_fraction"] = None

    values = extract_from_data(data, "ood_timeout_fraction")

    assert np.isnan(values[0])
    assert values[1] == pytest.approx(0.75)


def test_legacy_artifact_reports_missing_heist_metric():
    data = {
        "meta": np.array(
            [
                {
                    "summary": {
                        "test": {
                            "level_ood_gt": [False],
                            "level_ood_pred": [False],
                        }
                    }
                }
            ],
            dtype=object,
        )
    }

    with pytest.raises(
        ValueError,
        match="mean_oracle_regret.*unavailable at curve point 0.*heist_afh",
    ):
        extract_from_data(data, "mean_oracle_regret")


def test_heist_metric_plotting_keeps_lowest_order_duplicate_afhp():
    data, _, second = make_curve_data()
    data["meta"][1]["summary"]["test"]["level_ood_pred"] = [False, False]
    data["order"] = np.array([2, 1])

    x, y = extract_x_and_y_values(data, "level_afhp", "mean_oracle_regret")

    np.testing.assert_allclose(x, [0.0])
    np.testing.assert_allclose(y, [second["mean_oracle_regret"]])

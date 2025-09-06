# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import numpy as np


from analyzing.utils import eval_result_plotter


def extract_from_data(data, key: str) -> np.ndarray:
    if key == "ood_pred_percentage":
        return [element['summary']['test']['ood_pred_percentage'] for element in data['meta']]
    elif key == "ood_accuracy":
        return [element['summary']['test']['ood_accuracy'] for element in data['meta']]
    elif key == "performance":
        return data["performances"]
    elif key == "afhp":
        return data["afhps"]
    else:
        raise ValueError(f"Invalid key: {key}")


def extract_x_and_y_values(data, x_data_key: str, y_data_key: str) -> tuple[np.ndarray, np.ndarray]:
    # x = data["afhps"]
    x = extract_from_data(data, x_data_key)
    y = extract_from_data(data, y_data_key)
    return x, y


if __name__ == "__main__":
    eval_result_plotter(extract_x_and_y_values, "OOD Prediction Percentage")

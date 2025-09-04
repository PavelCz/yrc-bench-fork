# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import numpy as np


from analyzing.utils import eval_result_plotter



def extract_x_and_y_values(data) -> tuple[np.ndarray, np.ndarray]:
    # x = data["afhps"]
    x = [element['summary']['test']['ood_pred_percentage'] for element in data['meta']]
    y = data["performances"]
    return x, y


if __name__ == "__main__":
    eval_result_plotter(extract_x_and_y_values, "OOD Prediction Percentage")

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

from pathlib import Path

import numpy as np

from analyzing.utils import eval_result_plotter


def extract_results() -> dict[str, Path]:
    base_path = Path("/home/pavel/data/goal-misgen/tmp")
    eval_path = base_path / "28-defer-to-oracle"

    # prefix_filter = "24-easy-policy"
    prefix_filter = None

    evals = {}

    for child in eval_path.iterdir():
        if child.is_dir():
            method_name = child.name
            if (child / "eval_runs").exists():
                for grandchild in (child / "eval_runs").iterdir():
                    for grandgrandchild in grandchild.iterdir():
                        if (
                            grandgrandchild.is_file()
                            and grandgrandchild.suffix == ".npz"
                        ):
                            if prefix_filter is None or grandchild.stem.startswith(
                                f"eval-{prefix_filter}"
                            ):
                                evals[method_name] = grandgrandchild
    return evals


def extract_x_and_y_values(data) -> tuple[np.ndarray, np.ndarray]:
    x = data["afhps"]
    y = data["performances"]
    return x, y


if __name__ == "__main__":
    eval_result_plotter(extract_results, extract_x_and_y_values)

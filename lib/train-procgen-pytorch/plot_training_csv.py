#!/usr/bin/env python3
from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "plot_training_csv.py"
    runpy.run_path(str(script_path), run_name="__main__")

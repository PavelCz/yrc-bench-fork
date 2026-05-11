"""Probe whether a trained DeepSVDD checkpoint outputs the same decision score
for any input, or only for procgen-like inputs.

Loads the joblib checkpoint directly (no env / no YRC policy stack), then runs
``clf.decision_function`` on a battery of synthetic [N, 3, 64, 64] inputs:
zeros, ones, uniform/Gaussian noise, single-channel deltas, large positive
and negative values, NaN/inf. Prints the per-input score range and unique count.

If every synthetic input collapses to the same score the procgen inputs
produced (0.3200000226 for ``svdd_coinrun_image_exp0``), the network is a
constant function. If synthetic inputs produce a *range* of scores, the
collapse is theme-specific to procgen rather than a true network collapse.

Run on a GPU node (the checkpoint pickles cuda tensors). Example::

    python scripts/probe_image_svdd_synthetic.py \\
        /nas/ucb/czempin/data/goal-misgen/trained_svdd/neurips04/svdd_coinrun_image_exp0/trained.joblib
"""

import argparse
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from joblib import load


def _summary(name: str, scores: np.ndarray) -> str:
    scores = scores.astype(np.float64).flatten()
    if scores.size == 0:
        return f"{name:>32} | empty"
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return f"{name:>32} | all non-finite (n={scores.size})"
    unique_count = np.unique(scores).size
    return (
        f"{name:>32} | unique={unique_count:>4d} | "
        f"min={scores.min():>20.14g} | "
        f"max={scores.max():>20.14g} | "
        f"mean={scores.mean():>20.14g}"
    )


def _build_probes(shape, device) -> Dict[str, torch.Tensor]:
    probes: Dict[str, torch.Tensor] = {}

    probes["zeros"] = torch.zeros(shape, device=device)
    probes["all_half_normalized"] = torch.full(shape, 0.5, device=device)
    probes["all_one_normalized"] = torch.ones(shape, device=device)
    probes["all_127_raw"] = torch.full(shape, 127.0, device=device)
    probes["all_255_raw"] = torch.full(shape, 255.0, device=device)
    probes["uniform_0_to_255"] = torch.rand(shape, device=device) * 255.0
    probes["normal_mean128_std50"] = torch.randn(shape, device=device) * 50.0 + 128.0
    probes["minus_ones"] = -torch.ones(shape, device=device)
    probes["minus_thousand"] = torch.full(shape, -1000.0, device=device)
    probes["plus_million"] = torch.full(shape, 1e6, device=device)
    probes["minus_million"] = torch.full(shape, -1e6, device=device)
    probes["nans"] = torch.full(shape, float("nan"), device=device)
    probes["plus_inf"] = torch.full(shape, float("inf"), device=device)
    probes["minus_inf"] = torch.full(shape, float("-inf"), device=device)

    for ch in range(shape[1]):
        x = torch.zeros(shape, device=device)
        x[:, ch, :, :] = 100.0
        probes[f"channel_{ch}_only_100"] = x

    delta = torch.zeros(shape, device=device)
    delta[:, :, 0, 0] = 255.0
    probes["single_pixel_255"] = delta

    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--shape",
        nargs=4,
        type=int,
        default=[8, 3, 64, 64],
        metavar=("N", "C", "H", "W"),
    )
    args = parser.parse_args()

    state = load(args.model_path)
    clf = state["clf"]

    device = next(clf.model_.parameters()).device
    print(f"checkpoint: {args.model_path}")
    print(f"clf_name: {state.get('clf_name')}")
    print(f"model device: {device}")
    print(f"input_shape attr: {getattr(clf, 'input_shape', None)}")
    if hasattr(clf, "c") and clf.c is not None:
        c = clf.c
        c_flat = c.detach().flatten().cpu().numpy()
        print(
            f"center c: shape={tuple(c.shape)}, "
            f"norm={float(np.linalg.norm(c_flat)):.10g}, "
            f"head={c_flat[: min(6, c_flat.size)]}"
        )
    if hasattr(clf, "threshold_"):
        print(f"clf.threshold_: {clf.threshold_}")
    print(f"shape used for probes: {args.shape}")
    print()

    probes = _build_probes(tuple(args.shape), device)
    print("Synthetic probe scores:")
    print("-" * 110)
    for name, x in probes.items():
        try:
            scores = clf.decision_function(x)
            print(_summary(name, np.asarray(scores)))
        except Exception as exc:
            print(f"{name:>32} | ERROR: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

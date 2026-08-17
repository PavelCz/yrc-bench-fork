#!/usr/bin/env python3
"""Report novice/expert Heist proxy-trigger rates from AFHP artifacts."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EXPERIMENT_PATTERN = re.compile(r"^(.+)_heist_exp(\d+)$")
METHOD_PATTERN = re.compile(r"^heist_(.+)_exp(\d+)$")
ENDPOINTS = (("novice", np.argmin, 0.0), ("expert", np.argmax, 1.0))


@dataclass(frozen=True)
class EndpointTriggerRates:
    """OOD-only trigger rates for one acting-policy endpoint evaluation."""

    exp_id: int
    method: str
    role: str
    ood_episodes: int
    redundant_key_rate: float
    all_keys_rate: float
    artifact: Path


def parse_experiment_dir(name: str) -> Optional[Tuple[str, int]]:
    """Parse ``<prefix>_heist_exp<N>`` directory names."""
    match = EXPERIMENT_PATTERN.match(name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def parse_method_dir(name: str) -> Optional[Tuple[str, int]]:
    """Parse ``heist_<method>_exp<N>`` directory names."""
    match = METHOD_PATTERN.match(name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _latest_npz(method_dir: Path) -> Optional[Path]:
    """Return the test artifact from the lexicographically latest run."""
    for run_dir in sorted(
        method_dir.iterdir(), key=lambda path: path.name, reverse=True
    ):
        if not run_dir.is_dir():
            continue
        test_files = sorted(run_dir.glob("*_test.npz"))
        candidates = test_files or sorted(run_dir.glob("*.npz"))
        if candidates:
            return candidates[-1]
    return None


def discover_artifacts(
    eval_dir: Path,
    prefixes: Optional[Sequence[str]] = None,
    exp_ids: Optional[Sequence[int]] = None,
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
) -> Dict[Tuple[int, str], Path]:
    """Find the latest Heist artifact for each experiment and method."""
    if not eval_dir.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {eval_dir}")

    prefix_filter = set(prefixes) if prefixes is not None else None
    exp_filter = set(exp_ids) if exp_ids is not None else None
    include_filter = set(include_methods) if include_methods is not None else None
    exclude_filter = set(exclude_methods or [])
    artifacts: Dict[Tuple[int, str], Path] = {}
    latest_run_names: Dict[Tuple[int, str], str] = {}

    for experiment_dir in sorted(eval_dir.iterdir(), key=lambda path: path.name):
        if not experiment_dir.is_dir():
            continue
        parsed_experiment = parse_experiment_dir(experiment_dir.name)
        if parsed_experiment is None:
            continue
        prefix, exp_id = parsed_experiment
        if prefix_filter is not None and prefix not in prefix_filter:
            continue
        if exp_filter is not None and exp_id not in exp_filter:
            continue

        for method_dir in sorted(experiment_dir.iterdir(), key=lambda path: path.name):
            if not method_dir.is_dir():
                continue
            parsed_method = parse_method_dir(method_dir.name)
            if parsed_method is None:
                continue
            method, method_exp_id = parsed_method
            if method_exp_id != exp_id:
                continue
            if include_filter is not None and method not in include_filter:
                continue
            if method in exclude_filter:
                continue

            artifact = _latest_npz(method_dir)
            if artifact is None:
                continue
            key = (exp_id, method)
            run_name = artifact.parent.name
            if run_name > latest_run_names.get(key, ""):
                artifacts[key] = artifact
                latest_run_names[key] = run_name

    return artifacts


def _aligned_episode_arrays(summary: dict) -> Tuple[np.ndarray, ...]:
    fields = ("keys_collected", "num_keys", "total_chests", "level_ood_gt")
    missing = [field for field in fields if field not in summary]
    if missing:
        raise ValueError("Artifact summary is missing: " + ", ".join(missing))

    arrays = tuple(np.asarray(summary[field]) for field in fields)
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        details = ", ".join(
            f"{field}={len(values)}" for field, values in zip(fields, arrays)
        )
        raise ValueError(f"Artifact summary has misaligned episode arrays: {details}")
    return arrays


def load_endpoint_rates(
    artifact: Path, exp_id: int, method: str
) -> List[EndpointTriggerRates]:
    """Calculate OOD trigger rates at AFHP=0 and AFHP=1."""
    with np.load(artifact, allow_pickle=True) as data:
        if "afhps" not in data or "meta" not in data:
            raise ValueError(f"Artifact lacks AFHP curve data: {artifact}")
        afhps = np.asarray(data["afhps"], dtype=float)
        meta = data["meta"]
        if len(afhps) == 0 or len(afhps) != len(meta):
            raise ValueError(f"Artifact has invalid AFHP/meta arrays: {artifact}")

        results = []
        for role, endpoint_selector, expected_afhp in ENDPOINTS:
            point_index = int(endpoint_selector(afhps))
            actual_afhp = float(afhps[point_index])
            if not np.isclose(actual_afhp, expected_afhp):
                raise ValueError(
                    f"{artifact} has no AFHP={expected_afhp:g} {role} endpoint; "
                    f"closest value is {actual_afhp:g}"
                )

            summary = meta[point_index]["summary"]["test"]
            keys_collected, num_keys, total_chests, level_ood_gt = (
                _aligned_episode_arrays(summary)
            )
            ood_mask = level_ood_gt.astype(bool)
            ood_episodes = int(np.sum(ood_mask))
            if ood_episodes == 0:
                raise ValueError(f"{artifact} {role} endpoint has no OOD episodes")

            results.append(
                EndpointTriggerRates(
                    exp_id=exp_id,
                    method=method,
                    role=role,
                    ood_episodes=ood_episodes,
                    redundant_key_rate=float(
                        np.mean((keys_collected > total_chests)[ood_mask])
                    ),
                    all_keys_rate=float(
                        np.mean((keys_collected >= num_keys)[ood_mask])
                    ),
                    artifact=artifact,
                )
            )
    return results


def calculate_rates(
    artifacts: Dict[Tuple[int, str], Path],
) -> List[EndpointTriggerRates]:
    """Load all discovered artifacts and return endpoint trigger rates."""
    rows = []
    for (exp_id, method), artifact in sorted(artifacts.items()):
        rows.extend(load_endpoint_rates(artifact, exp_id, method))
    return rows


def experiment_means(
    rows: Iterable[EndpointTriggerRates],
) -> Dict[Tuple[int, str], Tuple[int, float, float]]:
    """Average duplicate method endpoints within each experiment and role."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.exp_id, row.role)].append(row)

    return {
        key: (
            len(group_rows),
            float(np.mean([row.redundant_key_rate for row in group_rows])),
            float(np.mean([row.all_keys_rate for row in group_rows])),
        )
        for key, group_rows in grouped.items()
    }


def median_iqr(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return median, 25th percentile, and 75th percentile."""
    values_np = np.asarray(values, dtype=float)
    return (
        float(np.median(values_np)),
        float(np.quantile(values_np, 0.25)),
        float(np.quantile(values_np, 0.75)),
    )


def print_report(rows: Sequence[EndpointTriggerRates], show_runs: bool = False) -> None:
    """Print Markdown tables of endpoint and aggregate trigger rates."""
    if show_runs:
        print("### Per-artifact OOD endpoint rates\n")
        print(
            "| Experiment | Method | Role | OOD episodes | "
            "Redundant-key trigger | All-keys trigger |"
        )
        print("|---:|:---|:---|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row.exp_id} | {row.method} | {row.role} | "
                f"{row.ood_episodes} | {row.redundant_key_rate:.3%} | "
                f"{row.all_keys_rate:.3%} |"
            )
        print()

    means = experiment_means(rows)
    print("### Per-experiment OOD trigger rates\n")
    print("Mean across method endpoint evaluations.\n")
    print("| Experiment | Role | Methods | Redundant-key trigger | All-keys trigger |")
    print("|---:|:---|---:|---:|---:|")
    for (exp_id, role), (method_count, redundant_rate, all_keys_rate) in sorted(
        means.items()
    ):
        print(
            f"| {exp_id} | {role} | {method_count} | "
            f"{redundant_rate:.3%} | {all_keys_rate:.3%} |"
        )

    print("\n### Aggregate OOD trigger rates\n")
    print("Median [IQR] across experiments.\n")
    print("| Role | Redundant-key trigger | All-keys trigger |")
    print("|:---|---:|---:|")
    for role in ("novice", "expert"):
        role_values = [
            values for (_, row_role), values in means.items() if row_role == role
        ]
        if not role_values:
            continue
        redundant = median_iqr([values[1] for values in role_values])
        all_keys = median_iqr([values[2] for values in role_values])
        print(
            f"| {role} | {redundant[0]:.3%} "
            f"[{redundant[1]:.3%}, {redundant[2]:.3%}] | "
            f"{all_keys[0]:.3%} [{all_keys[1]:.3%}, {all_keys[2]:.3%}] |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate OOD first-redundant-key and collect-all-keys trigger "
            "rates at the novice (AFHP=0) and expert (AFHP=1) endpoints."
        )
    )
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--prefix", nargs="+", default=None)
    parser.add_argument("--exp_ids", nargs="+", type=int, default=None)
    parser.add_argument("--method_include_filter", nargs="+", default=None)
    parser.add_argument("--method_filter", nargs="+", default=None)
    parser.add_argument(
        "--show_runs",
        action="store_true",
        help="Also print every method-by-experiment endpoint rate.",
    )
    args = parser.parse_args()

    artifacts = discover_artifacts(
        eval_dir=args.eval_dir,
        prefixes=args.prefix,
        exp_ids=args.exp_ids,
        include_methods=args.method_include_filter,
        exclude_methods=args.method_filter,
    )
    if not artifacts:
        parser.error("No matching Heist AFHP artifacts found")

    rows = calculate_rates(artifacts)
    print_report(rows, show_runs=args.show_runs)


if __name__ == "__main__":
    main()

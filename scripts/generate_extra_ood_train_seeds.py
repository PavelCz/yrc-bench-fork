#!/usr/bin/env python3
"""Generate an additional OOD-train seed file without editing existing files."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set


LEVEL_SEED_SPLITS = ("policy_train", "ood_train", "validation", "ood_eval")
DEFAULT_MIN_SEED = 100_000
DEFAULT_MAX_SEED = 2_147_483_647


def discover_level_seed_files(paths: List[Path]) -> List[Path]:
    """Return JSON seed files from explicit files or direct child files of dirs."""
    seed_files: List[Path] = []

    for path in paths:
        if path.is_dir():
            seed_files.extend(
                sorted(child for child in path.glob("*.json") if child.is_file())
            )
        elif path.is_file():
            seed_files.append(path)
        else:
            raise FileNotFoundError(f"Level seed path does not exist: {path}")

    unique_seed_files = list(dict.fromkeys(seed_files))
    if not unique_seed_files:
        raise ValueError("No level seed JSON files found to exclude.")

    return unique_seed_files


def load_excluded_seeds(seed_files: List[Path]) -> Set[int]:
    """Load every seed from every split in existing level seed files."""
    excluded: Set[int] = set()

    for seed_file in seed_files:
        with open(seed_file) as f:
            seed_data = json.load(f)

        split_data = seed_data.get("seeds")
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Level seed file is missing a 'seeds' object: {seed_file}"
            )

        for split_name, seeds in split_data.items():
            if seeds is None:
                continue
            if not isinstance(seeds, list):
                raise ValueError(
                    f"Split {split_name!r} in {seed_file} must be a list, got {type(seeds).__name__}."
                )

            for seed in seeds:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(
                        f"Split {split_name!r} in {seed_file} contains non-integer seed {seed!r}."
                    )
                excluded.add(seed)

    return excluded


def generate_ood_train_seeds(
    num_seeds: int,
    excluded_seeds: Set[int],
    *,
    base_seed: int,
    min_seed: int,
    max_seed: int,
) -> List[int]:
    """Generate unique OOD-train seeds excluding all existing seed values."""
    if num_seeds <= 0:
        raise ValueError("--ood-train must be positive.")
    if min_seed > max_seed:
        raise ValueError("--min-seed must be less than or equal to --max-seed.")

    excluded_in_range = {
        seed for seed in excluded_seeds if min_seed <= seed <= max_seed
    }
    available_count = max_seed - min_seed + 1 - len(excluded_in_range)
    if num_seeds > available_count:
        raise ValueError(
            f"Requested {num_seeds} OOD-train seeds, but only {available_count} "
            f"values are available in [{min_seed}, {max_seed}] after exclusions."
        )

    rng = random.Random(base_seed)
    selected: Set[int] = set()
    ordered_seeds: List[int] = []

    while len(ordered_seeds) < num_seeds:
        candidate = rng.randint(min_seed, max_seed)
        if candidate in excluded_seeds or candidate in selected:
            continue
        selected.add(candidate)
        ordered_seeds.append(candidate)

    return ordered_seeds


def build_seed_file_data(
    ood_train_seeds: List[int],
    *,
    source_files: List[Path],
    excluded_count: int,
    base_seed: int,
    min_seed: int,
    max_seed: int,
    name: str | None = None,
) -> Dict[str, Any]:
    seed_splits = {split: [] for split in LEVEL_SEED_SPLITS}
    seed_splits["ood_train"] = ood_train_seeds

    metadata: Dict[str, Any] = {
        "base_seed": base_seed,
        "num_policy_train": 0,
        "num_ood_train": len(ood_train_seeds),
        "num_validation": 0,
        "num_ood_eval": 0,
        "candidate_min_seed": min_seed,
        "candidate_max_seed": max_seed,
        "excluded_level_seed_count": excluded_count,
        "source_level_seed_files": [str(path) for path in source_files],
        "description": (
            "Additional OOD training level seeds generated from a separate "
            "candidate range while excluding all seeds found in the listed "
            "source level seed files. Intended for gather_rollouts.py only."
        ),
    }
    if name is not None:
        metadata["name"] = name

    return {
        "metadata": metadata,
        "seeds": seed_splits,
    }


def save_seed_file(
    seed_file_data: Dict[str, Any], output_path: Path, *, overwrite: bool
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(seed_file_data, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an OOD-train-only level seed file that does not overlap with "
            "existing level seed JSON files."
        )
    )
    parser.add_argument(
        "--existing-level-seeds",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Existing level seed JSON files or directories of JSON files. Every seed "
            "in every split is excluded from the generated OOD-train seeds."
        ),
    )
    parser.add_argument(
        "--ood-train",
        type=int,
        required=True,
        help="Number of additional OOD-train seeds to generate.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base RNG seed for reproducible generation (default: 0).",
    )
    parser.add_argument(
        "--min-seed",
        type=int,
        default=DEFAULT_MIN_SEED,
        help=f"Inclusive lower bound for generated seeds (default: {DEFAULT_MIN_SEED}).",
    )
    parser.add_argument(
        "--max-seed",
        type=int,
        default=DEFAULT_MAX_SEED,
        help=f"Inclusive upper bound for generated seeds (default: {DEFAULT_MAX_SEED}).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional human-readable name stored in metadata.name.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_files = discover_level_seed_files(args.existing_level_seeds)
    excluded_seeds = load_excluded_seeds(source_files)
    ood_train_seeds = generate_ood_train_seeds(
        args.ood_train,
        excluded_seeds,
        base_seed=args.base_seed,
        min_seed=args.min_seed,
        max_seed=args.max_seed,
    )
    seed_file_data = build_seed_file_data(
        ood_train_seeds,
        source_files=source_files,
        excluded_count=len(excluded_seeds),
        base_seed=args.base_seed,
        min_seed=args.min_seed,
        max_seed=args.max_seed,
        name=args.name,
    )
    save_seed_file(seed_file_data, args.output, overwrite=args.overwrite)

    print(f"Saved additional OOD-train seeds to {args.output}")
    if args.name is not None:
        print(f"  - Name: {args.name}")
    print(f"  - OOD training: {len(ood_train_seeds)} seeds")
    print(f"  - Excluded existing seeds: {len(excluded_seeds)}")
    print(f"  - Source files: {len(source_files)}")
    print(f"  - Candidate range: [{args.min_seed}, {args.max_seed}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

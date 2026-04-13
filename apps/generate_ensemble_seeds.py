#!/usr/bin/env python3
"""
Generate seed files for ensemble policy training.

This script creates multiple seed files for training an ensemble of policies,
where each policy is trained on a different set of level seeds. The generated
seeds do not overlap with any seeds in an existing seed file.

Usage:
    python -m apps.generate_ensemble_seeds -i seeds/0.json -o seeds/ensemble --num-seeds 100000 --num-members 4
    python -m apps.generate_ensemble_seeds -i seeds/0.json -o seeds/ensemble --num-seeds 100000 --num-members 4 --base-seed 42

The script will create files:
    seeds/ensemble_0.json
    seeds/ensemble_1.json
    seeds/ensemble_2.json
    seeds/ensemble_3.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Set

import numpy as np


def load_existing_seeds(input_path: Path) -> Set[int]:
    """
    Load all seeds from an existing seed file.

    Args:
        input_path: Path to the JSON seed file

    Returns:
        Set of all seeds used in the file
    """
    with open(input_path) as f:
        data = json.load(f)

    all_seeds: Set[int] = set()
    for phase, seeds in data["seeds"].items():
        all_seeds.update(seeds)

    return all_seeds


def generate_ensemble_seeds(
    num_seeds_per_member: int,
    num_members: int,
    excluded_seeds: Set[int],
    base_seed: int = 0,
) -> List[List[int]]:
    """
    Generate non-overlapping seed sets for ensemble members.

    Args:
        num_seeds_per_member: Number of seeds for each ensemble member
        num_members: Number of ensemble members
        excluded_seeds: Seeds to exclude (from existing seed file)
        base_seed: Base random seed for reproducible generation

    Returns:
        List of seed lists, one for each ensemble member
    """
    rng = np.random.default_rng(base_seed)

    # Calculate total seeds needed
    total_needed = num_seeds_per_member * num_members

    # Find the maximum excluded seed to determine our sampling range
    max_excluded = max(excluded_seeds) if excluded_seeds else 0

    # Create a pool of candidate seeds that don't overlap with excluded seeds
    # We sample from a range large enough to ensure we can get enough unique seeds
    # Start from max_excluded + 1 to avoid any overlap
    pool_start = max_excluded + 1
    pool_size = total_needed * 2  # Extra buffer for safety

    candidate_pool = np.arange(pool_start, pool_start + pool_size)
    rng.shuffle(candidate_pool)

    # Split into non-overlapping sets for each ensemble member
    ensemble_seeds: List[List[int]] = []
    for i in range(num_members):
        start_idx = i * num_seeds_per_member
        end_idx = start_idx + num_seeds_per_member
        member_seeds = candidate_pool[start_idx:end_idx].tolist()
        ensemble_seeds.append(member_seeds)

    return ensemble_seeds


def save_ensemble_seed_file(
    seeds: List[int],
    output_path: Path,
    member_idx: int,
    base_seed: int,
    original_seed_file: Path,
) -> None:
    """
    Save an ensemble member's seeds to a JSON file.

    The file format matches `apps/generate_level_seeds.py` output, with
    policy_train containing the ensemble seeds and empty lists for other phases.

    Args:
        seeds: List of seeds for this ensemble member
        output_path: Path to save the JSON file
        member_idx: Index of this ensemble member (0-indexed)
        base_seed: The base seed used for generation
        original_seed_file: Path to the original seed file (for documentation)
    """
    output_data = {
        "metadata": {
            "base_seed": base_seed,
            "ensemble_member": member_idx,
            "num_policy_train": len(seeds),
            "num_ood_train": 0,
            "num_validation": 0,
            "num_ood_eval": 0,
            "original_seed_file": str(original_seed_file),
            "description": (
                f"Ensemble member {member_idx} training seeds. "
                f"Generated to not overlap with seeds in {original_seed_file.name}. "
                "Use with level_seeds parameter in ProcgenGym3Env."
            ),
        },
        "seeds": {
            "policy_train": seeds,
            "ood_train": [],
            "validation": [],
            "ood_eval": [],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  - Member {member_idx}: {output_path} ({len(seeds)} seeds)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate seed files for ensemble policy training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 4 ensemble seed files with 100,000 seeds each
  python -m apps.generate_ensemble_seeds -i seeds/0.json -o seeds/ensemble --num-seeds 100000 --num-members 4

  # Use a specific base seed for reproducibility
  python -m apps.generate_ensemble_seeds -i seeds/0.json -o seeds/ensemble --num-seeds 100000 --num-members 4 --base-seed 42

The generated JSON files can be loaded and used with procgen:

  from apps.generate_level_seeds import load_seeds
  from procgen import ProcgenGym3Env

  seeds = load_seeds("seeds/ensemble_0.json")
  env = ProcgenGym3Env(num=4, env_name="coinrun", level_seeds=seeds["policy_train"])
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input seed file to avoid overlapping with",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        required=True,
        help="Output file prefix (e.g., 'seeds/ensemble' creates seeds/ensemble_0.json, etc.)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100_000,
        help="Number of seeds per ensemble member (default: 100000)",
    )
    parser.add_argument(
        "--num-members",
        type=int,
        default=4,
        help="Number of ensemble members (default: 4)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base random seed for deterministic generation (default: 0)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    if args.num_members <= 0:
        parser.error("--num-members must be positive")
    if not args.input.exists():
        parser.error(f"Input file does not exist: {args.input}")

    # Load existing seeds
    print(f"Loading existing seeds from {args.input}...")
    excluded_seeds = load_existing_seeds(args.input)
    print(f"  - Found {len(excluded_seeds)} existing seeds to exclude")

    # Generate ensemble seeds
    print(
        f"Generating {args.num_members} seed sets with {args.num_seeds} seeds each..."
    )
    ensemble_seeds = generate_ensemble_seeds(
        num_seeds_per_member=args.num_seeds,
        num_members=args.num_members,
        excluded_seeds=excluded_seeds,
        base_seed=args.base_seed,
    )

    # Save each ensemble member's seeds
    print("Saving ensemble seed files:")
    for i, seeds in enumerate(ensemble_seeds):
        output_path = Path(f"{args.output_prefix}_{i}.json")
        save_ensemble_seed_file(
            seeds=seeds,
            output_path=output_path,
            member_idx=i,
            base_seed=args.base_seed,
            original_seed_file=args.input,
        )

    print(f"\nGenerated {args.num_members} ensemble seed files.")
    print(f"Total unique seeds: {args.num_seeds * args.num_members}")


if __name__ == "__main__":
    main()

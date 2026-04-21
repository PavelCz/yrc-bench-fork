import json


def load_level_seed_splits(config, required_splits=()):
    """Load all non-empty level-seed splits from config.environment.level_seeds_file."""
    level_seeds_file = getattr(config.environment, "level_seeds_file", None)
    if level_seeds_file is None:
        if required_splits:
            required = ", ".join(sorted(required_splits))
            raise ValueError(
                "A level seeds file is required for seed-controlled splits: "
                f"{required}. Pass -level_seeds_file or do not require fixed seeds."
            )
        return {}

    print(f"Loading level seeds from {level_seeds_file}...")
    with open(level_seeds_file) as f:
        seeds_data = json.load(f)

    seed_splits = {
        split: seeds
        for split, seeds in seeds_data.get("seeds", {}).items()
        if seeds is not None and len(seeds) > 0
    }

    missing = [split for split in required_splits if split not in seed_splits]
    if missing:
        missing_str = ", ".join(sorted(missing))
        available = ", ".join(sorted(seed_splits)) or "none"
        raise ValueError(
            f"Level seeds file {level_seeds_file} is missing required split(s): "
            f"{missing_str}. Available non-empty splits: {available}."
        )

    if seed_splits:
        loaded = ", ".join(
            f"{split}={len(seeds)}" for split, seeds in sorted(seed_splits.items())
        )
        print(f"  - Loaded seed splits: {loaded}")
    else:
        print("  - No non-empty seed splits found")

    return seed_splits

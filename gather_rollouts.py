import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.level_seeds import load_level_seed_splits
from pathlib import Path
from YRC.core.rollout_helper import RolloutHelper
import torch
import json
import time


def select_rollout_level_seeds(all_level_seeds, requested_rollout_levels):
    if requested_rollout_levels is None:
        return list(all_level_seeds)
    if requested_rollout_levels <= 0:
        raise ValueError(
            f"rollout_levels must be positive when provided, got {requested_rollout_levels}"
        )
    if requested_rollout_levels > len(all_level_seeds):
        raise ValueError(
            f"Requested {requested_rollout_levels} rollout levels, but only "
            f"{len(all_level_seeds)} ood_train seeds are available."
        )
    return list(all_level_seeds[:requested_rollout_levels])


def build_rollout_paths(save_dir: Path, num_levels: int):
    suffix = f"{num_levels}levels"
    return {
        "config": save_dir / f"rollouts_config_{suffix}.json",
        "metadata": save_dir / f"rollouts_metadata_{suffix}.json",
        "data": save_dir / f"rollouts_{suffix}.pt",
    }


def main():
    total_start = time.time()

    print("Loading config...")
    start = time.time()
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)
    print(f"Config loaded in {time.time() - start:.2f}s")

    # Load all seed splits, then explicitly use ood_train for rollout collection.
    seed_splits = load_level_seed_splits(config, required_splits=("ood_train",))
    all_level_seeds = seed_splits["ood_train"]
    requested_rollout_levels = getattr(config, "rollout_levels", None)
    level_seeds = select_rollout_level_seeds(all_level_seeds, requested_rollout_levels)
    num_rollout_levels = len(level_seeds)

    print(
        f"Using {num_rollout_levels}/{len(all_level_seeds)} ood_train level seeds "
        "for rollout collection"
    )

    print("Creating environments...")
    start = time.time()
    envs = env_factory.make(
        config,
        level_seeds_by_split={"train": level_seeds},
        level_seeds_mode="sequential",
        require_level_seeds_for_splits=("train",),
    )
    print(f"Environments created in {time.time() - start:.2f}s")

    print(f"Gathering {num_rollout_levels} rollouts...")
    start = time.time()
    rollout_helper = RolloutHelper(config, envs["train"])
    rollout_obs, rollout_metadata = rollout_helper.gather_rollouts(
        envs["train"],
        num_rollout_levels,
        gather_all=True,
        return_list=True,
        return_metadata=True,
    )
    print(f"Rollouts gathered in {time.time() - start:.2f}s")

    save_dir = Path(str(get_global_variable("experiment_dir")))
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_paths = build_rollout_paths(save_dir, num_rollout_levels)

    print(f"Saving rollouts to {save_dir}")
    print(f"Rollout obs shape: {rollout_obs[0].shape}")

    # Save rollout obs to file.
    with rollout_paths["config"].open("w") as f:
        # Skip keys that are not JSON serializable, e.g. torch device.
        config_dict = config.as_dict()

        # Convert torch.device objects to strings for JSON serialization
        def convert_devices(obj):
            if isinstance(obj, dict):
                return {k: convert_devices(v) for k, v in obj.items()}
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_devices(item) for item in obj]
            else:
                return obj

        serializable_config = convert_devices(config_dict)
        json.dump(serializable_config, f)

    completed_level_seeds = rollout_metadata["completed_level_seeds"]
    if len(completed_level_seeds) != num_rollout_levels:
        raise ValueError(
            f"Expected {num_rollout_levels} completed rollout seeds, but recorded "
            f"{len(completed_level_seeds)}."
        )
    metadata_to_save = {
        "level_seeds_file": getattr(config.environment, "level_seeds_file", None),
        "total_available_ood_train_seeds": len(all_level_seeds),
        "requested_rollout_levels": requested_rollout_levels,
        "requested_level_seeds": level_seeds,
        "completed_level_seeds": completed_level_seeds,
        "num_requested_level_seeds": len(level_seeds)
        if level_seeds is not None
        else None,
        "num_completed_level_seeds": len(completed_level_seeds),
    }

    print(f"Saving rollout metadata to {rollout_paths['metadata']}")
    with rollout_paths["metadata"].open("w") as f:
        json.dump(metadata_to_save, f, indent=2)

    print(f"Saving rollouts to {rollout_paths['data']}")

    # Save rollout obs to file.
    start = time.time()
    with rollout_paths["data"].open("wb") as f:
        torch.save(rollout_obs, f)
    print(f"Rollouts saved in {time.time() - start:.2f}s")

    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()

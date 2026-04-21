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
    level_seeds = seed_splits["ood_train"]

    print("Creating environments...")
    start = time.time()
    envs = env_factory.make(
        config,
        level_seeds_by_split={"train": level_seeds},
        level_seeds_mode="sequential",
        require_level_seeds_for_splits=("train",),
    )
    print(f"Environments created in {time.time() - start:.2f}s")

    num_rollouts = config.algorithm.num_rollouts

    print(f"Gathering {num_rollouts} rollouts...")
    start = time.time()
    rollout_helper = RolloutHelper(config, envs["train"])
    rollout_obs, rollout_metadata = rollout_helper.gather_rollouts(
        envs["train"],
        num_rollouts,
        gather_all=True,
        return_list=True,
        return_metadata=True,
    )
    print(f"Rollouts gathered in {time.time() - start:.2f}s")

    save_dir = Path(str(get_global_variable("experiment_dir")))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving rollouts to {save_dir}")
    print(f"Rollout obs shape: {rollout_obs[0].shape}")

    # Save rollout obs to file.
    with (save_dir / "rollouts_config.json").open("w") as f:
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
    metadata_to_save = {
        "level_seeds_file": getattr(config.environment, "level_seeds_file", None),
        "requested_level_seeds": level_seeds,
        "completed_level_seeds": completed_level_seeds,
        "num_requested_level_seeds": len(level_seeds)
        if level_seeds is not None
        else None,
        "num_completed_level_seeds": len(completed_level_seeds),
    }

    print(f"Saving rollout metadata to {save_dir / 'rollouts_metadata.json'}")
    with (save_dir / "rollouts_metadata.json").open("w") as f:
        json.dump(metadata_to_save, f, indent=2)

    print(f"Saving rollouts to {save_dir / 'rollouts.pt'}")

    # Save rollout obs to file.
    start = time.time()
    with (save_dir / "rollouts.pt").open("wb") as f:
        torch.save(rollout_obs, f)
    print(f"Rollouts saved in {time.time() - start:.2f}s")

    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()

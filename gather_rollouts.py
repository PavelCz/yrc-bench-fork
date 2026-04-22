import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.level_seeds import load_level_seed_splits
from pathlib import Path
from YRC.core.rollout_storage import RolloutChunkWriter
from YRC.core.rollout_helper import RolloutHelper
import torch
import json
import time


DEFAULT_ROLLOUT_CHUNK_SIZE = 10_000


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
        "manifest": save_dir / f"rollouts_manifest_{suffix}.json",
        "chunks_dir": save_dir / f"rollouts_{suffix}_chunks",
    }


def get_rollout_chunk_size(config):
    rollout_chunk_size = getattr(config, "rollout_chunk_size", None)
    if rollout_chunk_size is None:
        return DEFAULT_ROLLOUT_CHUNK_SIZE
    if rollout_chunk_size <= 0:
        return None
    return rollout_chunk_size


def write_rollout_config(config, path: Path):
    with path.open("w") as f:
        serializable_config = config_utils.make_json_serializable(config.as_dict())
        json.dump(serializable_config, f)


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
    save_dir = Path(str(get_global_variable("experiment_dir")))
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_paths = build_rollout_paths(save_dir, num_rollout_levels)
    rollout_chunk_size = get_rollout_chunk_size(config)
    if rollout_chunk_size is not None and config.coord_policy.feature_type != "obs":
        print(
            "Chunked rollout saving currently supports feature_type='obs' only; "
            f"got {config.coord_policy.feature_type!r}, falling back to legacy save."
        )
        rollout_chunk_size = None

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
    chunk_writer = None
    if rollout_chunk_size is not None:
        print(
            "Chunked rollout saving enabled: writing at most "
            f"{rollout_chunk_size} observations per chunk"
        )
        chunk_writer = RolloutChunkWriter(
            rollout_paths["manifest"], rollout_paths["chunks_dir"]
        )
        rollout_obs, rollout_metadata = rollout_helper.gather_rollouts(
            envs["train"],
            num_rollout_levels,
            gather_all=True,
            return_list=True,
            return_metadata=True,
            chunk_size=rollout_chunk_size,
            chunk_callback=chunk_writer.write_chunk,
        )
    else:
        print("Chunked rollout saving disabled: keeping all observations in memory")
        rollout_obs, rollout_metadata = rollout_helper.gather_rollouts(
            envs["train"],
            num_rollout_levels,
            gather_all=True,
            return_list=True,
            return_metadata=True,
        )
    print(f"Rollouts gathered in {time.time() - start:.2f}s")

    print(f"Saving rollout config to {rollout_paths['config']}")
    write_rollout_config(config, rollout_paths["config"])

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
        "rollout_chunk_size": rollout_chunk_size,
    }
    if chunk_writer is not None:
        metadata_to_save.update(
            {
                "rollout_format": "chunked",
                "rollout_manifest": rollout_paths["manifest"].name,
                "num_rollout_chunks": chunk_writer.num_chunks,
                "num_rollout_observations": chunk_writer.num_observations,
                "first_observation_shape": chunk_writer.first_observation_shape,
            }
        )
    else:
        metadata_to_save.update(
            {
                "rollout_format": "legacy_pt",
                "rollout_data": rollout_paths["data"].name,
                "num_rollout_chunks": 0,
                "num_rollout_observations": len(rollout_obs),
                "first_observation_shape": list(rollout_obs[0].shape)
                if rollout_obs
                else None,
            }
        )

    print(f"Saving rollout metadata to {rollout_paths['metadata']}")
    with rollout_paths["metadata"].open("w") as f:
        json.dump(metadata_to_save, f, indent=2)

    start = time.time()
    if chunk_writer is not None:
        print(f"Saving rollout manifest to {rollout_paths['manifest']}")
        chunk_writer.save_manifest(
            {
                "feature_type": config.coord_policy.feature_type,
                "num_rollout_levels": num_rollout_levels,
                "rollout_chunk_size": rollout_chunk_size,
            }
        )
        print(
            "Rollout chunks saved: "
            f"{chunk_writer.num_chunks} chunks, "
            f"{chunk_writer.num_observations} observations"
        )
    else:
        print(f"Saving rollouts to {rollout_paths['data']}")
        with rollout_paths["data"].open("wb") as f:
            torch.save(rollout_obs, f)
    print(f"Rollouts saved in {time.time() - start:.2f}s")

    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()

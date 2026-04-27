import json
from pathlib import Path
from typing import List, Optional
import torch

from YRC.core.configs import ConfigDict
from YRC.core.rollout_storage import (
    IndexedChunkedRolloutDataset,
    MemmapRolloutDataset,
    is_rollout_chunk_manifest,
    is_rollout_memmap_metadata,
    load_chunked_rollouts,
)


def to_tensor(data):
    """Converts input to a torch tensor if it's not already."""
    if isinstance(data, dict):
        for key in data:
            data[key] = to_tensor(data[key])
        return data
    if isinstance(data, tuple):
        return data
    if not torch.is_tensor(data):
        # (pavel 2025-06-11) I removed the to device call since we don't want our
        # training dataset to be fully moved to the GPU by default. If this breaks
        # something somewhere else, I might have to reconsider this.
        return torch.from_numpy(data).float()  # .to(self.device)
    return data


def _print_dict_items(dictionary, color, indent):
    """Helper function to print dictionary items with specified color."""
    RESET = "\033[0m"
    indent_str = "  " * indent

    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if isinstance(value, dict):
            print(f"{indent_str}{color}  {key}:{RESET}")
            _print_dict_items(value, color, indent + 1)
        else:
            print(f"{indent_str}{color}  {key}: {value}{RESET}")


def print_dict_diff(
    dict1, dict2, dict1_name="Dict1", dict2_name="Dict2", print_output=True, indent=0
):
    """
    Compare two dictionaries and return differences, optionally printing them.

    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare
        dict1_name: Name for the first dictionary (for display)
        dict2_name: Name for the second dictionary (for display)
        print_output: Whether to print the differences (default: True)
        indent: Current indentation level for nested printing

    Returns:
        dict: Dictionary containing the differences with keys:
            - 'added': Items in dict2 but not in dict1
            - 'removed': Items in dict1 but not in dict2
            - 'changed': Items with different values between dict1 and dict2
    """
    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    indent_str = "  " * indent

    # Initialize result dictionary
    result = {"added": {}, "removed": {}, "changed": {}}

    # Keys only in dict1 (removed in dict2)
    removed_keys = set(dict1.keys()) - set(dict2.keys())

    # Keys only in dict2 (added in dict2)
    added_keys = set(dict2.keys()) - set(dict1.keys())

    # Keys in both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Handle removed keys
    for key in sorted(removed_keys):
        value = dict1[key]
        result["removed"][key] = value

        if print_output:
            if isinstance(value, dict):
                print(f"{indent_str}{RED}- {key}:{RESET}")
                _print_dict_items(value, RED, indent + 1)
            else:
                print(f"{indent_str}{RED}- {key}: {value}{RESET}")

    # Handle added keys
    for key in sorted(added_keys):
        value = dict2[key]
        result["added"][key] = value

        if print_output:
            if isinstance(value, dict):
                print(f"{indent_str}{GREEN}+ {key}:{RESET}")
                _print_dict_items(value, GREEN, indent + 1)
            else:
                print(f"{indent_str}{GREEN}+ {key}: {value}{RESET}")

    # Check common keys for value differences
    for key in sorted(common_keys):
        val1 = dict1[key]
        val2 = dict2[key]

        # If both values are dictionaries, recurse
        if isinstance(val1, dict) and isinstance(val2, dict):
            if val1 != val2:  # Only process if there are differences
                nested_diff = print_dict_diff(
                    val1, val2, dict1_name, dict2_name, print_output, indent + 1
                )

                # Only add to result if there are actual differences
                if (
                    nested_diff["added"]
                    or nested_diff["removed"]
                    or nested_diff["changed"]
                ):
                    result["changed"][key] = nested_diff

                    if print_output:
                        print(f"{indent_str}{key}:")

        # If values are different and not both dictionaries
        elif val1 != val2:
            result["changed"][key] = {"old": val1, "new": val2}

            if print_output:
                print(f"{indent_str}{key}:")
                print(f"{indent_str}  {RED}- {val1}{RESET}")
                print(f"{indent_str}  {GREEN}+ {val2}{RESET}")

    return result


def load_rollouts_from_file(
    rollout_dir: Path,
    config: Optional[ConfigDict] = None,
    max_levels: Optional[int] = None,
    prefer_largest: bool = False,
) -> List[torch.Tensor]:
    """Reads a ConfigDict and loads the rollouts, i.e. a dataset of collected
    observations from the file. As a sanity check the differences in the passed config
    and the config saved with the rollouts are printed.
    """
    rollouts_config_path, rollouts_data_path = resolve_rollout_paths(
        rollout_dir, prefer_largest=prefer_largest
    )
    print(f"Loading rollout dataset from {rollouts_data_path}")

    with rollouts_config_path.open("r") as f:
        rollouts_config_loaded = json.load(f)

    max_observations = _get_max_observations_for_levels(rollouts_data_path, max_levels)
    if max_levels is not None:
        print(
            f"Using first {max_levels} rollout levels"
            + (
                f" ({max_observations} observations)"
                if max_observations is not None
                else ""
            )
        )

    if is_rollout_chunk_manifest(rollouts_data_path):
        rollout_obs = load_chunked_rollouts(
            rollouts_data_path, max_observations=max_observations
        )
    else:
        with rollouts_data_path.open("rb") as f:
            rollout_obs = torch.load(f)
        if max_observations is not None:
            rollout_obs = rollout_obs[:max_observations]

    # for key, value in rollouts_config.items():
    #     if rollouts_config_loaded[key] != value:
    #         raise ValueError(
    #             f"Rollouts config mismatch: {rollouts_config_loaded[key]} != {value}"
    #         )

    if config is not None:
        # TODO: In the future, we can use the diff to check that certain important config
        # parameters are the same.
        print_dict_diff(config.as_dict(), rollouts_config_loaded)

    return rollout_obs


def load_rollout_dataset_from_file(
    rollout_dir: Path,
    config: Optional[ConfigDict] = None,
    max_levels: Optional[int] = None,
    prefer_largest: bool = False,
    streaming_rollouts: str = "auto",
    chunk_cache_size: int = 2,
):
    """Load rollout observations eagerly or as an indexed streaming dataset.

    Legacy ``.pt`` rollout artifacts are always loaded eagerly. Memmap rollout
    artifacts stream when available. Chunked rollout manifests stream when
    ``streaming_rollouts`` is ``"auto"`` or ``"true"``.
    """
    streaming_rollouts = _normalize_streaming_rollouts(streaming_rollouts)
    rollouts_config_path, rollouts_data_path = resolve_rollout_paths(
        rollout_dir, prefer_largest=prefer_largest
    )
    print(f"Loading rollout dataset from {rollouts_data_path}")

    with rollouts_config_path.open("r") as f:
        rollouts_config_loaded = json.load(f)

    max_observations = _get_max_observations_for_levels(rollouts_data_path, max_levels)
    if max_levels is not None:
        print(
            f"Using first {max_levels} rollout levels"
            + (
                f" ({max_observations} observations)"
                if max_observations is not None
                else ""
            )
        )

    should_stream = streaming_rollouts in {"auto", "true"}
    if is_rollout_memmap_metadata(rollouts_data_path):
        rollout_obs = MemmapRolloutDataset(
            rollouts_data_path,
            max_observations=max_observations,
        )
        print(f"Using memmap rollout dataset ({len(rollout_obs)} observations)")
    elif should_stream and is_rollout_chunk_manifest(rollouts_data_path):
        rollout_obs = IndexedChunkedRolloutDataset(
            rollouts_data_path,
            max_observations=max_observations,
            chunk_cache_size=chunk_cache_size,
        )
        print(
            "Using indexed streaming rollout dataset "
            f"({len(rollout_obs)} observations, chunk cache size {chunk_cache_size})"
        )
    elif is_rollout_chunk_manifest(rollouts_data_path):
        rollout_obs = load_chunked_rollouts(
            rollouts_data_path, max_observations=max_observations
        )
    else:
        with rollouts_data_path.open("rb") as f:
            rollout_obs = torch.load(f)
        if max_observations is not None:
            rollout_obs = rollout_obs[:max_observations]

    if config is not None:
        print_dict_diff(config.as_dict(), rollouts_config_loaded)

    return rollout_obs


def _normalize_streaming_rollouts(value) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "true" if value else "false"
    value = str(value).lower()
    if value not in {"auto", "true", "false"}:
        raise ValueError(
            "streaming_rollouts must be one of 'auto', 'true', or 'false', "
            f"got {value!r}"
        )
    return value


def resolve_rollout_paths(rollout_path: Path, prefer_largest: bool = False):
    rollout_path = Path(rollout_path)
    if rollout_path.is_file():
        return _resolve_rollout_file_paths(rollout_path)

    config_path = rollout_path / "rollouts_config.json"
    data_path = rollout_path / "rollouts.pt"
    if data_path.exists() and config_path.exists():
        return config_path, data_path

    matching_rollout_files = sorted(rollout_path.glob("rollouts_*levels.pt"))
    matching_manifests = sorted(rollout_path.glob("rollouts_manifest_*levels.json"))
    matching_memmaps = sorted(rollout_path.glob("rollouts_memmap_*levels.json"))
    matching_data_files = matching_rollout_files + matching_manifests + matching_memmaps
    if len(matching_data_files) == 1:
        return resolve_rollout_paths(matching_data_files[0])
    if len(matching_data_files) > 1:
        if prefer_largest:
            selected_path = max(matching_data_files, key=_rollout_selection_key)
            selected_levels = _rollout_level_count_from_path(selected_path)
            print(
                f"Multiple rollout datasets found in {rollout_path}; selecting "
                f"{selected_path.name} ({selected_levels} levels)."
            )
            return _resolve_rollout_file_paths(selected_path)
        available = ", ".join(path.name for path in matching_data_files)
        raise ValueError(
            f"Multiple rollout datasets found in {rollout_path}: {available}. "
            "Pass the specific rollout .pt file or rollout manifest you want to load."
        )

    if data_path.exists() != config_path.exists():
        missing = config_path if data_path.exists() else data_path
        raise FileNotFoundError(
            f"Incomplete legacy rollout dataset in {rollout_path}; missing {missing}."
        )

    raise FileNotFoundError(
        f"Could not find rollout dataset in {rollout_path}. Expected rollouts.pt "
        "or a single rollouts_*levels.pt / rollouts_manifest_*levels.json file."
    )


def _resolve_rollout_file_paths(rollout_path: Path):
    data_path = Path(rollout_path)
    if data_path.suffix == ".pt":
        if data_path.stem.startswith("rollouts_"):
            suffix = data_path.stem[len("rollouts_") :]
            config_path = data_path.with_name(f"rollouts_config_{suffix}.json")
        else:
            config_path = data_path.with_name("rollouts_config.json")
    elif is_rollout_chunk_manifest(data_path):
        suffix = data_path.stem[len("rollouts_manifest_") :]
        config_path = data_path.with_name(f"rollouts_config_{suffix}.json")
    elif is_rollout_memmap_metadata(data_path):
        suffix = data_path.stem[len("rollouts_memmap_") :]
        config_path = data_path.with_name(f"rollouts_config_{suffix}.json")
    else:
        raise ValueError(
            "Expected rollout file to be a .pt file, rollouts_manifest*.json, "
            "or rollouts_memmap*.json, "
            f"got {data_path}"
        )
    return config_path, data_path


def _rollout_level_count_from_path(path: Path) -> int:
    path = Path(path)
    if path.suffix == ".pt" and path.stem.startswith("rollouts_"):
        suffix = path.stem[len("rollouts_") :]
    elif is_rollout_chunk_manifest(path):
        suffix = path.stem[len("rollouts_manifest_") :]
    elif is_rollout_memmap_metadata(path):
        suffix = path.stem[len("rollouts_memmap_") :]
    else:
        raise ValueError(f"Cannot infer rollout level count from {path}")

    if not suffix.endswith("levels"):
        raise ValueError(f"Cannot infer rollout level count from {path}")
    level_count_str = suffix[: -len("levels")]
    try:
        return int(level_count_str)
    except ValueError as exc:
        raise ValueError(f"Cannot infer rollout level count from {path}") from exc


def _rollout_selection_key(path: Path):
    # Prefer larger artifacts, and for the same level count prefer memmap over
    # chunked manifest over legacy pt because memmap supports efficient random access.
    if is_rollout_memmap_metadata(path):
        format_priority = 2
    elif is_rollout_chunk_manifest(path):
        format_priority = 1
    else:
        format_priority = 0
    return _rollout_level_count_from_path(path), format_priority


def _get_rollout_metadata_path(data_path: Path) -> Path:
    data_path = Path(data_path)
    if data_path.suffix == ".pt" and data_path.stem.startswith("rollouts_"):
        suffix = data_path.stem[len("rollouts_") :]
    elif is_rollout_chunk_manifest(data_path):
        suffix = data_path.stem[len("rollouts_manifest_") :]
    elif is_rollout_memmap_metadata(data_path):
        suffix = data_path.stem[len("rollouts_memmap_") :]
    else:
        return data_path.with_name("rollouts_metadata.json")
    return data_path.with_name(f"rollouts_metadata_{suffix}.json")


def _load_rollout_level_observation_counts(data_path: Path) -> Optional[List[int]]:
    metadata_path = _get_rollout_metadata_path(data_path)
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        counts = metadata.get("completed_rollout_observation_counts")
        if counts is not None:
            return [int(count) for count in counts]

    if is_rollout_chunk_manifest(data_path):
        with Path(data_path).open("r") as f:
            manifest = json.load(f)
        counts = manifest.get("metadata", {}).get(
            "completed_rollout_observation_counts"
        )
        if counts is not None:
            return [int(count) for count in counts]
    if is_rollout_memmap_metadata(data_path):
        with Path(data_path).open("r") as f:
            metadata = json.load(f)
        counts = metadata.get("metadata", {}).get(
            "completed_rollout_observation_counts"
        )
        if counts is not None:
            return [int(count) for count in counts]
    return None


def _get_max_observations_for_levels(
    data_path: Path, max_levels: Optional[int]
) -> Optional[int]:
    if max_levels is None:
        return None
    if max_levels <= 0:
        raise ValueError(f"rollout max levels must be positive, got {max_levels}")

    artifact_levels = _rollout_level_count_from_path(data_path)
    if max_levels > artifact_levels:
        raise ValueError(
            f"Requested rollout max levels {max_levels}, but selected artifact "
            f"{data_path} only contains {artifact_levels} levels."
        )
    if max_levels == artifact_levels:
        return None

    observation_counts = _load_rollout_level_observation_counts(data_path)
    if observation_counts is None:
        raise ValueError(
            f"Cannot select the first {max_levels} levels from {data_path} because "
            "the rollout metadata does not contain per-level observation counts. "
            "Regather rollouts with the current gather_rollouts.py or pass a "
            "rollout artifact with exactly the requested level count."
        )
    if len(observation_counts) < max_levels:
        raise ValueError(
            f"Cannot select the first {max_levels} levels from {data_path}; metadata "
            f"only contains {len(observation_counts)} per-level counts."
        )
    return sum(observation_counts[:max_levels])

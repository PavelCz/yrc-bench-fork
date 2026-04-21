import json
from pathlib import Path
from typing import List, Optional
import torch

from YRC.core.configs import ConfigDict


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
    rollout_dir: Path, config: Optional[ConfigDict] = None
) -> List[torch.Tensor]:
    """Reads a ConfigDict and loads the rollouts, i.e. a dataset of collected
    observations from the file. As a sanity check the differences in the passed config
    and the config saved with the rollouts are printed.
    """
    rollouts_config_path, rollouts_data_path = resolve_rollout_paths(rollout_dir)

    with rollouts_config_path.open("r") as f:
        rollouts_config_loaded = json.load(f)

    with rollouts_data_path.open("rb") as f:
        rollout_obs = torch.load(f)

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


def resolve_rollout_paths(rollout_path: Path):
    rollout_path = Path(rollout_path)
    if rollout_path.is_file():
        if rollout_path.suffix != ".pt":
            raise ValueError(f"Expected rollout file to end in .pt, got {rollout_path}")
        data_path = rollout_path
        if rollout_path.stem.startswith("rollouts_"):
            suffix = rollout_path.stem[len("rollouts_") :]
            config_path = rollout_path.with_name(f"rollouts_config_{suffix}.json")
        else:
            config_path = rollout_path.with_name("rollouts_config.json")
        return config_path, data_path

    config_path = rollout_path / "rollouts_config.json"
    data_path = rollout_path / "rollouts.pt"
    if data_path.exists() and config_path.exists():
        return config_path, data_path

    matching_rollout_files = sorted(rollout_path.glob("rollouts_*levels.pt"))
    if len(matching_rollout_files) == 1:
        return resolve_rollout_paths(matching_rollout_files[0])
    if len(matching_rollout_files) > 1:
        available = ", ".join(path.name for path in matching_rollout_files)
        raise ValueError(
            f"Multiple rollout datasets found in {rollout_path}: {available}. "
            "Pass the specific rollout .pt file you want to load."
        )

    raise FileNotFoundError(
        f"Could not find rollout dataset in {rollout_path}. Expected rollouts.pt "
        "or a single rollouts_*levels.pt file."
    )

import json
from pathlib import Path
from typing import List
import torch

from YRC.core.configs import ConfigDict
from YRC.core.configs.global_configs import get_global_variable

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
    RESET = '\033[0m'
    indent_str = "  " * indent
    
    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if isinstance(value, dict):
            print(f"{indent_str}{color}  {key}:{RESET}")
            _print_dict_items(value, color, indent + 1)
        else:
            print(f"{indent_str}{color}  {key}: {value}{RESET}")


def print_dict_diff(dict1, dict2, dict1_name="Dict1", dict2_name="Dict2", print_output=True, indent=0):
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
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    indent_str = "  " * indent
    
    # Initialize result dictionary
    result = {
        'added': {},
        'removed': {},
        'changed': {}
    }
    
    # Keys only in dict1 (removed in dict2)
    removed_keys = set(dict1.keys()) - set(dict2.keys())
    
    # Keys only in dict2 (added in dict2)  
    added_keys = set(dict2.keys()) - set(dict1.keys())
    
    # Keys in both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    # Handle removed keys
    for key in sorted(removed_keys):
        value = dict1[key]
        result['removed'][key] = value
        
        if print_output:
            if isinstance(value, dict):
                print(f"{indent_str}{RED}- {key}:{RESET}")
                _print_dict_items(value, RED, indent + 1)
            else:
                print(f"{indent_str}{RED}- {key}: {value}{RESET}")
    
    # Handle added keys
    for key in sorted(added_keys):
        value = dict2[key]
        result['added'][key] = value
        
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
                nested_diff = print_dict_diff(val1, val2, dict1_name, dict2_name, print_output, indent + 1)
                
                # Only add to result if there are actual differences
                if nested_diff['added'] or nested_diff['removed'] or nested_diff['changed']:
                    result['changed'][key] = nested_diff
                    
                    if print_output:
                        print(f"{indent_str}{key}:")
        
        # If values are different and not both dictionaries
        elif val1 != val2:
            result['changed'][key] = {'old': val1, 'new': val2}
            
            if print_output:
                print(f"{indent_str}{key}:")
                print(f"{indent_str}  {RED}- {val1}{RESET}")
                print(f"{indent_str}  {GREEN}+ {val2}{RESET}")
    
    return result


def load_rollouts_from_file(config: ConfigDict) -> List[torch.Tensor]:
    """Reads a ConfigDict and loads the rollouts, i.e. a dataset of collected
    observations from the file. As a sanity check the differences in the passed config
    and the config saved with the rollouts are printed.
    """
    experiment_dir = Path(str(get_global_variable("experiment_dir")))

    output_dir = experiment_dir.parent
    rollouts_dir = output_dir / config.training.rollout_dir

    with (rollouts_dir / "rollouts_config.json").open("r") as f:
        rollouts_config_loaded = json.load(f)

    with (rollouts_dir / "rollouts.pt").open("rb") as f:
        rollout_obs = torch.load(f)

    # for key, value in rollouts_config.items():
    #     if rollouts_config_loaded[key] != value:
    #         raise ValueError(
    #             f"Rollouts config mismatch: {rollouts_config_loaded[key]} != {value}"
    #         )

    print(f"Loaded rollouts from {rollouts_dir}")
    print(f"Rollout obs shape: {rollout_obs[0].shape}")
    # print(f"Number of rollouts: {rollouts_config['num_rollouts']}")

    # TODO: In the future, we can use the diff to check that certain important config
    # parameters are the same.
    diff = print_dict_diff(config.as_dict(), rollouts_config_loaded)

    return rollout_obs
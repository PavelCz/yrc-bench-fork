import torch

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


def print_dict_diff(dict1, dict2, dict1_name="Dict1", dict2_name="Dict2", indent=0):
    """
    Print differences between two dictionaries with colored output.
    
    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare  
        dict1_name: Name for the first dictionary (for display)
        dict2_name: Name for the second dictionary (for display)
        indent: Current indentation level for nested printing
    """
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    indent_str = "  " * indent
    
    # Get all keys from both dictionaries
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    # Keys only in dict1 (removed in dict2)
    removed_keys = set(dict1.keys()) - set(dict2.keys())
    
    # Keys only in dict2 (added in dict2)  
    added_keys = set(dict2.keys()) - set(dict1.keys())
    
    # Keys in both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    # Print removed keys (red)
    for key in sorted(removed_keys):
        value = dict1[key]
        if isinstance(value, dict):
            print(f"{indent_str}{RED}- {key}:{RESET}")
            _print_dict_items(value, RED, indent + 1)
        else:
            print(f"{indent_str}{RED}- {key}: {value}{RESET}")
    
    # Print added keys (green)  
    for key in sorted(added_keys):
        value = dict2[key]
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
            if val1 != val2:  # Only print if there are differences
                print(f"{indent_str}{key}:")
                print_dict_diff(val1, val2, dict1_name, dict2_name, indent + 1)
        
        # If values are different and not both dictionaries
        elif val1 != val2:
            print(f"{indent_str}{key}:")
            print(f"{indent_str}  {RED}- {val1}{RESET}")
            print(f"{indent_str}  {GREEN}+ {val2}{RESET}")
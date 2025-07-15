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
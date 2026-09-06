import numpy as np
import random
import gym
import torch
import torch.nn as nn


def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_global_log_levels(level):
    gym.logger.set_level(level)


def orthogonal_init(module, gain=nn.init.calculate_gain("relu")):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def get_n_params(model):
    return (
        str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3))
        + " M params"
    )


def episode_randomize_goal(info, done, current_value):
    """Return the OOD label for the current or just-finished episode.

    Procgen auto-resets on done, so terminal infos expose the finished
    episode as ``prev_level/randomize_goal``.
    """
    if done and "prev_level/randomize_goal" in info:
        return bool(info["prev_level/randomize_goal"])
    return bool(current_value)


def compute_oracle_regret(episode_return, num_keys, total_chests):
    """Return oracle-normalized regret for a zero-key-penalty Heist episode.

    Must stay aligned with ``YRC.envs.procgen.heist_metrics.compute_oracle_regret``.
    """
    cap = min(num_keys, total_chests)
    if cap <= 0:
        raise ValueError(
            "Heist oracle regret requires a positive reward cap, got "
            f"num_keys={num_keys}, total_chests={total_chests}"
        )
    return 1.0 - float(episode_return) / float(cap)


def heist_oracle_regret_from_info(episode_return, info):
    """Return Heist oracle regret from terminal info, or None if it is undefined.

    Training validation can finish an episode with ``num_keys=0`` and
    ``total_chests=0`` (empty prev-level counters). Skip those rather than
    raising; eval still errors on a non-positive cap.
    """
    if "prev_level/num_keys" not in info or "prev_level/total_chests" not in info:
        return None
    num_keys = int(info["prev_level/num_keys"])
    total_chests = int(info["prev_level/total_chests"])
    if min(num_keys, total_chests) <= 0:
        return None
    return compute_oracle_regret(episode_return, num_keys, total_chests)

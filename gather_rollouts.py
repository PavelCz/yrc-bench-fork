import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
from YRC.core.configs.global_configs import get_global_variable
from pathlib import Path
from YRC.core.rollout_helper import RolloutHelper
from typing import List
import torch
import json


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)

    num_rollouts = config.algorithm.num_rollouts

    rollout_helper = RolloutHelper(config, envs["train"])
    rollout_obs: List[torch.Tensor] = rollout_helper.gather_rollouts(
        envs["train"], num_rollouts, gather_all=True, return_list=True
    )

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

    print(f"Saving rollouts to {save_dir / 'rollouts.pt'}")

    # Save rollout obs to file.
    with (save_dir / "rollouts.pt").open("wb") as f:
        torch.save(rollout_obs, f)


if __name__ == "__main__":
    main()
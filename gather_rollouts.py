import flags
import YRC.core.algorithm as algo_factory
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
import wandb
from pytorch_lightning.loggers import WandbLogger
from YRC.core.configs.global_configs import get_global_variable
from pathlib import Path
from YRC.core.rollout_helper import RolloutHelper
from typing import List
import torch
import numpy as np
import json


def main():
    args = flags.make()
    config = config_utils.load_partial(args.config, "gather_rollouts", flags=args)

    envs = env_factory.make(config)

    num_rollouts = config.num_rollouts

    rollout_helper = RolloutHelper(config, envs["train"])
    rollout_obs: List[torch.Tensor] = rollout_helper.gather_rollouts(
        envs["train"], num_rollouts, gather_all=True, return_list=True
    )

    rollouts_config = {
        "num_rollouts": num_rollouts,
        "feature_type": config.feature_type,
        "collect_data_agent": config.collect_data_agent,
    }

    rollouts_dir = Path(config.rollout_dir)
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving rollouts to {rollouts_dir}")
    print(f"Rollout obs shape: {rollout_obs[0].shape}")

    # Save rollout obs to file.
    with (rollouts_dir / "rollouts_config.json").open("w") as f:
        json.dump(rollouts_config, f)

    print(f"Saving rollouts to {rollouts_dir / 'rollouts.pt'}")

    # Save rollout obs to file.
    with (rollouts_dir / "rollouts.pt").open("wb") as f:
        torch.save(rollout_obs, f)


if __name__ == "__main__":
    main()
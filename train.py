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
import json
import torch
from typing import List
from YRC.core.configs import ConfigDict
from YRC.core.utils import print_dict_diff

# Algorithms that support training without threshold search.
ALGORITHMS = ["ood", "lightning_ae"]

def main():
    args = flags.make()
    config: ConfigDict = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    evaluator = Evaluator(config.evaluation)

    if config.training.rollout_dir is not None:
        rollouts = load_rollouts(config)
        policy.load_rollouts(rollouts)

    if hasattr(policy, "logger"):

        save_dir = Path(str(get_global_variable("experiment_dir")))

        # Prepare wandb init parameters
        wandb_kwargs = {
            "name": config.exp_name,
            "project": config.wandb.project,
            "group": config.wandb.group,
            "mode": config.wandb.mode,
            "job_type": "train",
            "config": config,
        }

        if config.wandb.entity is not None:
            wandb_kwargs["entity"] = config.wandb.entity

        exp = wandb.init(**wandb_kwargs)

        wandb_logger = WandbLogger(
            save_dir=save_dir, experiment=exp,
        )
        policy.logger = wandb_logger

    if config.general.algorithm not in ALGORITHMS:
        raise NotImplementedError(
            f"Algorithm {config.general.algorithm} does not support training without "
            "threshold search."
        )

    algorithm = algo_factory.make(config, envs["train"])
    algorithm.train(
        policy,
        envs,
        evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
        do_threshold_search=False,
    )


def load_rollouts(config: ConfigDict) -> List[torch.Tensor]:

    rollouts_dir = Path(config.training.rollout_dir)

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

    print_dict_diff(config.as_dict(), rollouts_config_loaded)

    return rollout_obs


if __name__ == "__main__":
    main()
from YRC.core.utils import load_rollouts_from_file
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
from YRC.core.configs import ConfigDict
from typing import List
import torch

# Algorithms that support training without threshold search.
ALGORITHMS = ["ood", "lightning_ae"]

def main():
    args = flags.make()
    config: ConfigDict = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    evaluator = Evaluator(config)

    if config.training.rollout_dir is not None:

        experiment_dir = Path(str(get_global_variable("experiment_dir")))

        output_dir = experiment_dir.parent
        rollout_dir = output_dir / config.training.rollout_dir
        rollout_obs = load_rollouts_from_file(rollout_dir, config)

    ae_inputs: List[torch.Tensor]
    if config.coord_policy.feature_type == "obs":
        ae_inputs = rollout_obs
    elif config.coord_policy.feature_type == "hidden":
        ae_inputs = []
        # Query weak agent to get hidden features from observations.
        weak_agent = envs["train"].weak_agent
        weak_agent.eval()
        for obs in rollout_obs:
            obs = obs.to(weak_agent.model.device).unsqueeze(0)
            hidden_features = weak_agent.get_hidden(obs).detach().cpu()
            ae_inputs.append(hidden_features)
    else:
        raise ValueError(
            f"Feature type {config.coord_policy.feature_type} currently not supported"
        )

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
        policy=policy,
        envs=envs,
        rollout_obs=ae_inputs,
        evaluator=evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
        do_threshold_search=False,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
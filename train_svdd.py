"""Train OOD/SVDD-style coordination policies from collected rollouts."""

from YRC.core.utils import load_rollouts_from_file
from YRC.core.level_seeds import load_level_seed_splits
from YRC.core.rollout_helper import RolloutHelper
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
from types import SimpleNamespace
from typing import List, Optional
import torch
import time
from gather_rollouts import make_gather_env_and_agent

# Algorithms that support training without threshold search.
ALGORITHMS = ["ood", "lightning_ae"]


def select_level_seed_subset(all_level_seeds, requested_levels, split_name):
    if requested_levels is None:
        return None
    if requested_levels <= 0:
        raise ValueError(
            f"{split_name} level count must be positive when provided, got "
            f"{requested_levels}"
        )
    if requested_levels > len(all_level_seeds):
        raise ValueError(
            f"Requested {requested_levels} {split_name} levels, but only "
            f"{len(all_level_seeds)} seeds are available."
        )
    return list(all_level_seeds[:requested_levels])


def collect_svdd_validation_rollouts(config):
    training_config = getattr(config, "training", None)
    svdd_val_levels = getattr(training_config, "svdd_val_levels", None)
    if svdd_val_levels is None:
        return None

    seed_splits = load_level_seed_splits(config, required_splits=("validation",))
    level_seeds = select_level_seed_subset(
        seed_splits["validation"], svdd_val_levels, "SVDD validation"
    )

    print(
        f"Collecting SVDD validation rollouts on {len(level_seeds)} validation levels..."
    )
    start = time.time()
    gather_env, gather_agent = make_gather_env_and_agent(config, level_seeds)
    rollout_config = SimpleNamespace(coord_policy=SimpleNamespace(feature_type="obs"))
    rollout_helper = RolloutHelper(rollout_config, gather_env, agent=gather_agent)
    try:
        rollout_obs, metadata = rollout_helper.gather_acting_policy_rollouts(
            gather_env,
            len(level_seeds),
            gather_all=True,
            return_metadata=True,
        )
    finally:
        close = getattr(gather_env, "close", None)
        if close is not None:
            close()

    completed_level_seeds = metadata["completed_level_seeds"]
    if len(completed_level_seeds) != len(level_seeds):
        raise RuntimeError(
            "SVDD validation rollout collection did not complete the requested "
            f"levels. Requested {len(level_seeds)}, completed "
            f"{len(completed_level_seeds)}."
        )

    print(f"SVDD validation rollouts collected in {time.time() - start:.2f}s")
    return rollout_obs


def process_rollout_features(config, envs, rollout_obs) -> Optional[List[torch.Tensor]]:
    if rollout_obs is None:
        return None

    if config.coord_policy.feature_type == "obs":
        return rollout_obs
    if config.coord_policy.feature_type == "hidden":
        inputs = []
        # Query weak agent to get hidden features from observations.
        weak_agent = envs["train"].weak_agent
        weak_agent.eval()
        with torch.no_grad():
            for obs in rollout_obs:
                obs = obs.to(weak_agent.model.device).unsqueeze(0)
                hidden_features = weak_agent.get_hidden(obs).detach().cpu()
                inputs.append(hidden_features)
        return inputs
    raise ValueError(
        f"Feature type {config.coord_policy.feature_type} currently not supported"
    )


def main():
    total_start = time.time()

    print("Loading config...")
    start = time.time()
    args = flags.make()
    config: ConfigDict = config_utils.load(args.config, flags=args)
    print(f"Config loaded in {time.time() - start:.2f}s")

    print("Creating environments...")
    start = time.time()
    envs = env_factory.make(config)
    print(f"Environments created in {time.time() - start:.2f}s")

    print("Creating policy...")
    start = time.time()
    policy = policy_factory.make(config, envs["train"])
    print(f"Policy created in {time.time() - start:.2f}s")

    print("Creating evaluator...")
    start = time.time()
    evaluator = Evaluator(config)
    print(f"Evaluator created in {time.time() - start:.2f}s")

    if config.training.rollout_dir is not None:
        print("Loading rollouts...")
        start = time.time()
        experiment_dir = Path(str(get_global_variable("experiment_dir")))

        output_dir = experiment_dir.parent
        rollout_dir = output_dir / config.training.rollout_dir
        rollout_max_levels = getattr(config.training, "rollout_max_levels", None)
        rollout_obs = load_rollouts_from_file(
            rollout_dir,
            config,
            max_levels=rollout_max_levels,
            prefer_largest=True,
        )
        print(f"Rollouts loaded in {time.time() - start:.2f}s")

    val_rollout_obs = collect_svdd_validation_rollouts(config)

    print("Processing features...")
    start = time.time()
    ae_inputs = process_rollout_features(config, envs, rollout_obs)
    val_inputs = process_rollout_features(config, envs, val_rollout_obs)
    print(f"Features processed in {time.time() - start:.2f}s")

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
            save_dir=save_dir,
            experiment=exp,
        )
        policy.logger = wandb_logger

    if config.general.algorithm not in ALGORITHMS:
        raise NotImplementedError(
            f"Algorithm {config.general.algorithm} does not support training without "
            "threshold search."
        )

    print("Starting training...")
    start = time.time()
    algorithm = algo_factory.make(config, envs["train"])
    algorithm.train(
        policy=policy,
        envs=envs,
        rollout_obs=ae_inputs,
        val_rollout_obs=val_inputs,
        evaluator=evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
        do_threshold_search=False,
    )
    print(f"Training completed in {time.time() - start:.2f}s")
    print(f"Total time: {time.time() - total_start:.2f}s")
    wandb.finish()


if __name__ == "__main__":
    main()

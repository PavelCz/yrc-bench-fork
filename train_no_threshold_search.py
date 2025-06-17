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

# Algorithms that support training without threshold search.
ALGORITHMS = ["ood", "lightning_ae"]

if __name__ == "__main__":
    args = flags.make()
    config = config_utils.load(args.config, flags=args)


    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    evaluator = Evaluator(config.evaluation)

    if hasattr(policy, "logger"):

        save_dir = Path(str(get_global_variable("experiment_dir")))

        exp = wandb.init(
            name=config.exp_name,
            project="yrc-01",
            group=config.exp_name,
            mode="online",
            job_type="train",
            config=config,
        )

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

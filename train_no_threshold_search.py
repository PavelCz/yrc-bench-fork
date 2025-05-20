import flags
import YRC.core.algorithm as algo_factory
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator

# Algorithms that support training without threshold search.
ALGORITHMS = ["ood"]

if __name__ == "__main__":
    args = flags.make()
    config = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    evaluator = Evaluator(config.evaluation)

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

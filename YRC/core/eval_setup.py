from dataclasses import dataclass
import json
import os
from typing import Any, Callable, Dict, List, Optional

import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy


NON_MODEL_ALGORITHMS = {
    "timestep_random",
    "level_based_random",
    "threshold",
    "heuristic",
    "wait",
}


@dataclass
class EvalRuntime:
    config: Any
    policy: Any
    evaluator: Evaluator
    envs: Dict[str, Any]
    make_envs: Callable[[], Dict[str, Any]]
    ood_eval_seeds: Optional[List[int]]
    cal_seeds: Optional[List[int]]

    def close_envs(self) -> None:
        for split_name in self.envs:
            self.envs[split_name].close()


def load_level_seeds(config) -> Dict[str, Optional[List[int]]]:
    """Load fixed evaluation/calibration seeds from the configured JSON file."""
    level_seeds_file = getattr(config.environment, "level_seeds_file", None)
    if level_seeds_file is None:
        return {"ood_eval": None, "validation": None}

    print(f"Loading level seeds from {level_seeds_file}...")
    with open(level_seeds_file) as f:
        seeds_data = json.load(f)

    ood_eval = seeds_data["seeds"].get("ood_eval") or None
    validation = seeds_data["seeds"].get("validation") or None

    if ood_eval:
        print(f"  Loaded {len(ood_eval)} ood_eval seeds")
    if validation:
        print(f"  Loaded {len(validation)} validation seeds (calibration)")

    return {"ood_eval": ood_eval, "validation": validation}


def build_eval_runtime(config) -> EvalRuntime:
    """Build the shared runtime used by AFHP evaluation entrypoints."""
    seeds = load_level_seeds(config)
    ood_eval_seeds = seeds["ood_eval"]
    cal_seeds = seeds["validation"]

    def make_envs():
        return env_factory.make(
            config,
            ood_eval_seeds,
            "sequential",
            cal_seeds=cal_seeds,
        )

    envs = make_envs()
    policy = policy_factory.make(config, envs["train"])

    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        if config.general.algorithm not in NON_MODEL_ALGORITHMS:
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config, config.environment)
    return EvalRuntime(
        config=config,
        policy=policy,
        evaluator=evaluator,
        envs=envs,
        make_envs=make_envs,
        ood_eval_seeds=ood_eval_seeds,
        cal_seeds=cal_seeds,
    )

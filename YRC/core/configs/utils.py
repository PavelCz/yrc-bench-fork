import os
import sys
import logging
import time
import traceback
import wandb
import json

import yaml
from datetime import datetime
from pathlib import Path

import torch
import numpy as np


from YRC.core.configs import ConfigDict
from YRC.core.configs.global_configs import set_global_variable


def make_json_serializable(obj):
    """Convert config dict values to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_config_json(config: ConfigDict, output_path: Path):
    """Save config as JSON file for later filtering and grouping."""
    config_dict = config.as_dict()
    serializable_config = make_json_serializable(config_dict)

    with open(output_path, "w") as f:
        json.dump(serializable_config, f, indent=2, sort_keys=True)


def load(yaml_file_or_str, flags=None) -> ConfigDict:
    if yaml_file_or_str.endswith(".yaml"):
        with open(yaml_file_or_str) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = yaml.safe_load(yaml_file_or_str)

    with open("configs/common.yaml") as f:
        common_config = yaml.safe_load(f)
        common_config = ConfigDict(**common_config)

    config = ConfigDict(**config_dict)
    algorithm = config.general.algorithm
    benchmark = config.general.benchmark
    if config.coord_policy is None:
        config.coord_policy = getattr(common_config.coord_policy, algorithm)
    if config.coord_env is None:
        config.coord_env = getattr(common_config.coord_env, benchmark)
    if config.evaluation is None:
        config.evaluation = getattr(common_config.evaluation, benchmark)
    if config.algorithm is None:
        config.algorithm = getattr(common_config.algorithm, algorithm)
    if config.environment is None:
        config.environment = getattr(common_config.environment, benchmark)

    if flags is not None:
        config_dict = config.as_dict()
        flags_dict = flags.as_dict()

        # Handle experiment_group special logic
        if (
            "experiment_group" in flags_dict
            and flags_dict["experiment_group"] is not None
        ):
            experiment_group = flags_dict["experiment_group"]

            # Set wandb group if not already set
            if "wandb" not in flags_dict:
                flags_dict["wandb"] = {}
            if (
                "group" not in flags_dict["wandb"]
                or flags_dict["wandb"]["group"] is None
            ):
                flags_dict["wandb"]["group"] = experiment_group

            # Set eval_run_name as prefix if not already set
            if "eval_run_name" not in flags_dict or flags_dict["eval_run_name"] is None:
                flags_dict["eval_run_name"] = experiment_group

        update_config(flags_dict, config_dict)
        config = ConfigDict(**config_dict)

    # Only copy env_name_suffix for environments that use it (e.g., minigrid)
    if (
        hasattr(config.environment.train, "env_name_suffix")
        and config.environment.val_sim is not None
    ):
        config.environment.val_sim.env_name_suffix = (
            config.environment.train.env_name_suffix
        )
    if (
        hasattr(config.environment.test, "env_name_suffix")
        and config.environment.val_true is not None
    ):
        config.environment.val_true.env_name_suffix = (
            config.environment.test.env_name_suffix
        )

    config.data_dir = os.getenv("SM_DATA_DIR", config.data_dir)
    output_dir = Path(os.getenv("SM_OUTPUT_DIR", "experiments"))
    if config.name is None:
        raise ValueError("config.name is None. A name must be provided.")
    config.experiment_dir = str(output_dir / config.name)

    if not config.eval_mode and (config.overwrite is None or not config.overwrite):
        try:
            Path(config.experiment_dir).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                "Experiment directory %s probably exists!" % config.experiment_dir
            )
    Path(config.experiment_dir).mkdir(parents=True, exist_ok=True)

    seed = config.general.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # config.random = random.Random(seed)

    if config.general.device == -1:
        config.general.device = torch.device("cpu")
    else:
        config.general.device = torch.device("cuda", config.general.device)
    set_global_variable("device", config.general.device)
    set_global_variable("benchmark", config.general.benchmark)
    set_global_variable("experiment_dir", config.experiment_dir)
    set_global_variable("seed", config.general.seed)

    config.start_time = time.time()

    if config.eval_mode:
        # Require experiment_group to be set for eval mode
        experiment_group = None
        if hasattr(config, "experiment_group") and config.experiment_group:
            experiment_group = config.experiment_group
        elif (
            hasattr(config, "wandb")
            and hasattr(config.wandb, "group")
            and config.wandb.group
        ):
            experiment_group = config.wandb.group

        if not experiment_group:
            raise ValueError(
                "experiment_group must be set for eval mode. "
                "Use --experiment_group flag or --wandb.group flag."
            )

        # Generate timestamp for eval run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model name from experiment_dir (last component of path)
        model_name = Path(config.experiment_dir).name

        # Build new eval directory structure:
        # experiments/evals/<experiment_group>/<model_name>/<timestamp>/
        output_dir = Path(os.getenv("SM_OUTPUT_DIR", "experiments"))
        eval_run_dir = output_dir / "evals" / experiment_group / model_name / timestamp

        # Create the directory if it doesn't exist
        eval_run_dir.mkdir(parents=True, exist_ok=True)

        # Save config as JSON for later filtering and grouping
        config_json_path = eval_run_dir / "config.json"
        save_config_json(config, config_json_path)

        # Set log file path based on file_name type
        if config.file_name is None or "trained" in config.file_name:
            log_file = str(eval_run_dir / f"eval_seed_{seed}.log")
        elif config.file_name.__contains__("sim"):
            log_file = str(eval_run_dir / f"eval_sim_seed_{seed}.log")
        elif config.file_name.__contains__("true"):
            log_file = str(eval_run_dir / f"eval_true_seed_{seed}.log")
        else:
            raise ValueError(
                f"Unrecognized eval setting with file name: {config.file_name}"
            )

        # Store eval run directory in config for potential use by other components
        config.eval_run_dir = str(eval_run_dir)
    else:
        log_file = str(Path(config.experiment_dir) / "run.log")
    set_global_variable("log_file", log_file)

    log_file_path = Path(log_file)
    if log_file_path.is_file():
        log_file_path.unlink()

    # Get log level from flags if available, otherwise default to INFO
    log_level = logging.INFO
    if (
        flags is not None
        and hasattr(flags, "log_level")
        and flags.log_level is not None
    ):
        log_level = getattr(logging, flags.log_level)

    config_logging(log_file, log_level=log_level)
    logging.info(str(datetime.now()))
    logging.info("python -u " + " ".join(sys.argv))
    logging.info("Write log to %s" % log_file)
    if config.eval_mode:
        config_json_path = Path(config.eval_run_dir) / "config.json"
        logging.info("Saved eval config to %s" % config_json_path)
    logging.info(str(config))

    wandb.init(
        project="YRC",
        name=f"{config.name}_{str(int(time.time()))}",
        mode="online" if config.use_wandb else "disabled",
    )
    wandb.config.update(config)
    return config


def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]


def config_logging(log_file, log_level=logging.INFO):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(
        level=log_level, handlers=[file_handler, stream_handler], force=True
    )

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler


class ElapsedFormatter(logging.Formatter):
    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "[%s %s]: %s" % (elapsed_time, record.levelname, record.getMessage())
        return log_str

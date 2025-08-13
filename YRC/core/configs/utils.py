import os
import sys
import logging
import time
import traceback
import wandb

import yaml
from datetime import datetime

import torch
import numpy as np


from YRC.core.configs import ConfigDict
from YRC.core.configs.global_configs import set_global_variable


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
        if "experiment_group" in flags_dict and flags_dict["experiment_group"] is not None:
            experiment_group = flags_dict["experiment_group"]
            
            # Set wandb group if not already set
            if "wandb" not in flags_dict:
                flags_dict["wandb"] = {}
            if "group" not in flags_dict["wandb"] or flags_dict["wandb"]["group"] is None:
                flags_dict["wandb"]["group"] = experiment_group
            
            # Set eval_run_name as prefix if not already set
            if "eval_run_name" not in flags_dict or flags_dict["eval_run_name"] is None:
                flags_dict["eval_run_name"] = experiment_group
        
        update_config(flags_dict, config_dict)
        config = ConfigDict(**config_dict)

    config.environment.val_sim.env_name_suffix = config.environment.train.env_name_suffix
    config.environment.val_true.env_name_suffix = config.environment.test.env_name_suffix

    config.data_dir = os.getenv("SM_DATA_DIR", config.data_dir)
    output_dir = os.getenv("SM_OUTPUT_DIR", "experiments")
    config.experiment_dir = "%s/%s" % (output_dir, config.name)

    if not config.eval_mode and (config.overwrite is None or not config.overwrite):
        try:
            os.makedirs(config.experiment_dir)
        except:
            raise FileExistsError(
                "Experiment directory %s probably exists!" % config.experiment_dir
            )
    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)

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
        # Generate timestamp for eval run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create eval run name with timestamp
        if hasattr(config, 'eval_run_name') and config.eval_run_name:
            eval_dir_name = f"{config.eval_run_name}_{timestamp}"
        else:
            eval_dir_name = f"eval_{timestamp}"
        
        # Create eval runs subdirectory
        eval_runs_dir = os.path.join(config.experiment_dir, "eval_runs")
        eval_run_dir = os.path.join(eval_runs_dir, eval_dir_name)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(eval_run_dir):
            os.makedirs(eval_run_dir)
        
        # Set log file path based on file_name type
        if config.file_name is None or "trained" in config.file_name:
            log_file = os.path.join(eval_run_dir, f"eval_seed_{seed}.log")
        elif config.file_name.__contains__("sim"):
            log_file = os.path.join(eval_run_dir, f"eval_sim_seed_{seed}.log")
        elif config.file_name.__contains__("true"):
            log_file = os.path.join(eval_run_dir, f"eval_true_seed_{seed}.log")
        else:
            raise ValueError(
                f"Unrecognized eval setting with file name: {config.file_name}"
            )
        
        # Store eval run directory in config for potential use by other components
        config.eval_run_dir = eval_run_dir
    else:
        log_file = os.path.join(config.experiment_dir, "run.log")
    set_global_variable("log_file", log_file)

    if os.path.isfile(log_file):
        os.remove(log_file)
    config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info("python -u " + " ".join(sys.argv))
    logging.info("Write log to %s" % log_file)
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


def config_logging(log_file):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(
        level=logging.INFO, handlers=[file_handler, stream_handler], force=True
    )

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler


class ElapsedFormatter:
    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "[%s %s]: %s" % (elapsed_time, record.levelname, record.getMessage())
        return log_str
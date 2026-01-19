#!/usr/bin/env python3
"""
Script to run evaluation jobs in parallel via SLURM sbatch.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


# Default conda environment
DEFAULT_CONDA_ENV = "ood-stable"

# SLURM configuration
SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "3-00:00:00",
    "mem": "100G",
    "cpus-per-task": "30",
}

# Default evaluation configuration
EVAL_DEFAULTS = {
    "video_episodes_to_collect": 16,
    "num_levels": 5000,
    "video_filter": "all",
    "cp_rolling_average": "none",
    "video_logging_mode": "wandb",
    "video_filter_mode": "any",
}

# Server-specific paths
SERVER_PATHS = {
    "chai": {
        "checkpoint_base": "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        "seeds_base": "/nas/ucb/czempin/data/goal-misgen/seeds/icml",
        "svdd_base": "/nas/ucb/czempin/data/goal-misgen/trained_svdd",
    },
    "snoopy": {
        "checkpoint_base": "/scr/pavel/data/goal-misgen/policy/icml",
        "seeds_base": "/scr/pavel/data/goal-misgen/seeds/icml",
        "svdd_base": "/scr/pavel/data/goal-misgen/trained_svdd",
    },
}

# Environment choices
ENVS = ["maze", "coinrun"]

# Method to config file mapping
METHOD_CONFIGS = {
    "max-prob": "max_prob.yaml",
    "max-logit": "max_logit.yaml",
    "lb-random": "level_based_random.yaml",
    "ts-random": "timestep_random.yaml",
    "svdd-image": "image_svdd.yaml",
    "svdd-latent": "latent_svdd.yaml",
}

# Method to run name suffix mapping
METHOD_NAMES = {
    "max-prob": "max_prob",
    "max-logit": "max_logit",
    "lb-random": "lb_random",
    "ts-random": "ts_random",
    "svdd-image": "svdd_image",
    "svdd-latent": "svdd_latent",
}

# Methods that require a trained SVDD policy
SVDD_METHODS = {"svdd-image", "svdd-latent"}


def get_env_folder(env: str) -> str:
    """Get the environment folder name."""
    if env == "coinrun":
        return "coinrun"
    else:
        return f"{env}_afh"


def get_svdd_feature_type(method: str) -> str:
    """Get the SVDD feature type from method name."""
    if method == "svdd-image":
        return "image"
    elif method == "svdd-latent":
        return "latent"
    return ""


EXPECTED_TIMESTEPS = 200015872


def find_newest_timestamp_dir(parent_dir: Path) -> Optional[Path]:
    """Find the newest timestamp directory in parent_dir.

    Looks for dirs matching format: YYYY-MM-DD__HH-MM-SS__seed_*
    Returns the newest one based on the timestamp in the name.
    Prints a warning if multiple exist.
    """
    if not parent_dir.exists():
        return None

    # Find all timestamp directories
    timestamp_dirs = []
    for d in parent_dir.iterdir():
        if d.is_dir() and "__seed_" in d.name:
            timestamp_dirs.append(d)

    if not timestamp_dirs:
        return None

    if len(timestamp_dirs) > 1:
        print(f"Warning: Multiple timestamp dirs in {parent_dir}, using newest:")
        for d in sorted(timestamp_dirs, key=lambda x: x.name):
            print(f"  - {d.name}")

    # Sort by name (timestamp format sorts lexicographically)
    newest = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
    return newest


def find_best_model_checkpoint(ts_dir: Path) -> Optional[Path]:
    """Find the model checkpoint with highest timesteps.

    Looks for files matching format: model_*.pth
    Returns the one with highest timesteps number.
    Prints a warning if highest is not EXPECTED_TIMESTEPS.
    """
    if not ts_dir.exists():
        return None

    model_files = []
    for f in ts_dir.iterdir():
        if f.is_file() and f.name.startswith("model_") and f.name.endswith(".pth"):
            match = re.match(r"model_(\d+)\.pth", f.name)
            if match:
                timesteps = int(match.group(1))
                model_files.append((timesteps, f))

    if not model_files:
        return None

    # Sort by timesteps and get the highest
    model_files.sort(key=lambda x: x[0])
    highest_timesteps, best_model = model_files[-1]

    if highest_timesteps != EXPECTED_TIMESTEPS:
        print(f"Warning: {ts_dir.name} has max timesteps {highest_timesteps}, expected {EXPECTED_TIMESTEPS}")

    return best_model


def get_checkpoints(env: str, exp_id: int, checkpoint_base_path: str) -> dict:
    """Get checkpoint paths based on environment and experiment ID."""
    env_folder = get_env_folder(env)
    base_path = Path(checkpoint_base_path) / env_folder

    weak_parent = base_path / f"icml2_{env}_exp{exp_id}_0p"
    strong_parent = base_path / f"icml2_{env}_exp{exp_id}_50p"

    weak_ts_dir = find_newest_timestamp_dir(weak_parent)
    strong_ts_dir = find_newest_timestamp_dir(strong_parent)

    weak_model = find_best_model_checkpoint(weak_ts_dir) if weak_ts_dir else None
    strong_model = find_best_model_checkpoint(strong_ts_dir) if strong_ts_dir else None

    weak = str(weak_model) if weak_model else str(weak_parent / "NOT_FOUND")
    strong = str(strong_model) if strong_model else str(strong_parent / "NOT_FOUND")

    return {"sim": weak, "weak": weak, "strong": strong}


def get_svdd_policy_name(env: str, exp_id: int, method: str) -> str:
    """Get the SVDD policy directory name."""
    feature_type = get_svdd_feature_type(method)
    # Format: svdd_{env}_{feature_type}_exp{id}
    return f"svdd_{env}_{feature_type}_exp{exp_id}"


def get_svdd_model_path(env: str, exp_id: int, method: str, svdd_base_path: str) -> Optional[str]:
    """Get the full path to the trained SVDD model file, or None if it doesn't exist."""
    policy_name = get_svdd_policy_name(env, exp_id, method)
    model_file = Path(svdd_base_path) / policy_name / "trained.joblib"
    if model_file.exists():
        return str(model_file)
    return None


def build_sbatch_command(job_name: str, eval_args: dict, conda_env: str) -> str:
    """Build the sbatch command string."""
    slurm_args = " ".join(f"--{k}={v}" for k, v in SLURM_CONFIG.items())

    eval_cmd_parts = [
        f"conda run -n {conda_env} -- python eval_afhp.py",
        f"-c {eval_args['config']}",
        f"-n {eval_args['name']}",
        "-defer_to_oracle",
        f"-experiment_group {eval_args['experiment_group']}",
        f"-video_episodes_to_collect {eval_args['video_episodes_to_collect']}",
        f"-num_levels={eval_args['num_levels']}",
        f"-video_filter {eval_args['video_filter']}",
        f"-cp_rolling_average {eval_args['cp_rolling_average']}",
        f"-video_logging_mode={eval_args['video_logging_mode']}",
        f"-video_filter_mode={eval_args['video_filter_mode']}",
        f"-sim {eval_args['sim']}",
        f"-weak {eval_args['weak']}",
        f"-strong {eval_args['strong']}",
        f"-level_seeds_file {eval_args['level_seeds_file']}",
    ]

    # Add SVDD-specific arguments if present
    if eval_args.get('cp_feature'):
        eval_cmd_parts.append(f"-cp_feature {eval_args['cp_feature']}")
    if eval_args.get('svdd_model_path'):
        eval_cmd_parts.append(f"-f_n {eval_args['svdd_model_path']}")

    eval_cmd = " \\\n        ".join(eval_cmd_parts)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items())}

srun {slurm_args} \\
    {eval_cmd}
"""
    return sbatch_script


def submit_job(job_name: str, eval_args: dict, conda_env: str, dry_run: bool = False) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_command(job_name, eval_args, conda_env)

    if dry_run:
        print(f"=== Job: {job_name} ===")
        print(sbatch_script)
        print()
        return

    # Ensure logs directory exists
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)

    # Submit via sbatch
    result = subprocess.run(
        ["sbatch"],
        input=sbatch_script,
        text=True,
        capture_output=True,
    )

    if result.returncode == 0:
        print(f"Submitted {job_name}: {result.stdout.strip()}")
    else:
        print(f"Failed to submit {job_name}: {result.stderr}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation jobs via SLURM")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help=f"Conda environment to use (default: {DEFAULT_CONDA_ENV})")
    parser.add_argument("--server", choices=["chai", "snoopy"], default="chai", help="Server to use for paths (default: chai)")
    parser.add_argument("--env", required=True, choices=ENVS, help="Environment to evaluate")
    parser.add_argument("--method", required=True, choices=list(METHOD_CONFIGS.keys()), help="Evaluation method")
    parser.add_argument("--prefix", required=True, help="Experiment group prefix")
    parser.add_argument("--exp-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Experiment IDs to run (default: 0 1 2 3 4)")
    parser.add_argument("--num-levels", type=int, default=EVAL_DEFAULTS["num_levels"], help="Number of levels")
    parser.add_argument("--video-episodes", type=int, default=EVAL_DEFAULTS["video_episodes_to_collect"], help="Video episodes to collect")
    parser.add_argument("--video-filter", default=EVAL_DEFAULTS["video_filter"], help="Video filter")
    parser.add_argument("--cp-rolling-average", default=EVAL_DEFAULTS["cp_rolling_average"], help="Coordination policy rolling average")
    parser.add_argument("--video-logging-mode", default=EVAL_DEFAULTS["video_logging_mode"], help="Video logging mode")
    parser.add_argument("--video-filter-mode", default=EVAL_DEFAULTS["video_filter_mode"], help="Video filter mode")
    # Override checkpoints if needed
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    args = parser.parse_args()

    # Get server-specific paths
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    seeds_base_path = paths["seeds_base"]
    svdd_base_path = paths["svdd_base"]

    # Get config file path
    config_file = METHOD_CONFIGS[args.method]
    config_path = f"configs/eval/{args.env}/{config_file}"

    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {config_path}")
        print(f"Environment: {args.env}")
        print(f"Method: {args.method}")
        print(f"Prefix: {args.prefix}")
        print()

    # Loop over experiment IDs
    for exp_id in args.exp_ids:
        # Get checkpoints for this experiment
        checkpoints = get_checkpoints(args.env, exp_id, checkpoint_base_path)
        if args.sim:
            checkpoints["sim"] = args.sim
        if args.weak:
            checkpoints["weak"] = args.weak
        if args.strong:
            checkpoints["strong"] = args.strong

        # Get level seeds file path
        level_seeds_file = Path(seeds_base_path) / f"{exp_id}.json"

        # Get SVDD-specific settings if needed
        svdd_model_path = None
        cp_feature = None
        if args.method in SVDD_METHODS:
            svdd_model_path = get_svdd_model_path(args.env, exp_id, args.method, svdd_base_path)
            cp_feature = "obs" if args.method == "svdd-image" else "hidden"

        # Validate checkpoints and seeds file exist
        missing = False
        for name, path in checkpoints.items():
            if not Path(path).exists():
                print(f"Warning: exp{exp_id} {name} checkpoint not found: {path}")
                missing = True

        if not level_seeds_file.exists():
            print(f"Warning: exp{exp_id} level seeds file not found: {level_seeds_file}")
            missing = True

        if args.method in SVDD_METHODS and svdd_model_path is None:
            svdd_policy_name = get_svdd_policy_name(args.env, exp_id, args.method)
            print(f"Warning: exp{exp_id} SVDD model not found at {svdd_base_path}/{svdd_policy_name}/trained.joblib")
            missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing files\n")
            continue

        # Build job name and experiment group
        method_name = METHOD_NAMES[args.method]
        job_name = f"{args.env}_{method_name}_exp{exp_id}"
        experiment_group = f"{args.prefix}_{args.env}_exp{exp_id}"

        if args.dry_run:
            print(f"=== exp{exp_id} ===")
            print(f"  Job name: {job_name}")
            print(f"  Experiment group: {experiment_group}")
            print(f"  Weak:   {checkpoints['weak']}")
            print(f"  Strong: {checkpoints['strong']}")
            print(f"  Seeds:  {level_seeds_file}")
            if svdd_model_path:
                print(f"  SVDD model: {svdd_model_path}")
                print(f"  Feature type: {cp_feature}")
            print()
            continue

        eval_args = {
            "config": config_path,
            "name": job_name,
            "experiment_group": experiment_group,
            "num_levels": args.num_levels,
            "video_episodes_to_collect": args.video_episodes,
            "video_filter": args.video_filter,
            "cp_rolling_average": args.cp_rolling_average,
            "video_logging_mode": args.video_logging_mode,
            "video_filter_mode": args.video_filter_mode,
            "level_seeds_file": str(level_seeds_file),
            "svdd_model_path": svdd_model_path,
            "cp_feature": cp_feature,
            **checkpoints,
        }

        submit_job(job_name, eval_args, args.conda_env, dry_run=False)

    return 0


if __name__ == "__main__":
    exit(main())

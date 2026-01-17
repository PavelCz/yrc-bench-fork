#!/usr/bin/env python3
"""
Script to run evaluation jobs in parallel via SLURM sbatch.
"""

import os
import subprocess
from pathlib import Path


# SLURM configuration
SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "3-00:00:00",
    "mem": "256G",
    "cpus-per-task": "32",
}

# Default evaluation configuration
EVAL_DEFAULTS = {
    "video_episodes_to_collect": 16,
    "num_levels": 10240,
    "video_filter": "all",
    "cp_rolling_average": "none",
    "video_logging_mode": "wandb",
    "video_filter_mode": "any",
}

# Base path for checkpoints
CHECKPOINT_BASE_PATH = "/nas/ucb/czempin/data/goal-misgen/policy/icml"

# Environment choices
ENVS = ["maze", "coinrun"]

# Method to config file mapping
METHOD_CONFIGS = {
    "max-prob": "threshold.yaml",
    "lb-random": "level_based_random.yaml",
    "ts-random": "timestep_random.yaml",
}

# Method to run name suffix mapping
METHOD_NAMES = {
    "max-prob": "max_prob",
    "lb-random": "lb_random",
    "ts-random": "ts_random",
}


def get_env_folder(env: str) -> str:
    """Get the environment folder name."""
    if env == "coinrun":
        return "coinrun"
    else:
        return f"{env}_afh"


def get_checkpoints(env: str, exp_id: int) -> dict:
    """Get checkpoint paths based on environment and experiment ID."""
    env_folder = get_env_folder(env)
    base_path = f"{CHECKPOINT_BASE_PATH}/{env_folder}"
    weak = f"{base_path}/icml2_{env}_exp{exp_id}_0p/weak.pt"
    strong = f"{base_path}/icml2_{env}_exp{exp_id}_50p/strong.pt"
    return {"sim": weak, "weak": weak, "strong": strong}


def build_sbatch_command(job_name: str, eval_args: dict) -> str:
    """Build the sbatch command string."""
    slurm_args = " ".join(f"--{k}={v}" for k, v in SLURM_CONFIG.items())

    eval_cmd_parts = [
        "python eval_afhp.py",
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
    ]
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


def submit_job(job_name: str, eval_args: dict, dry_run: bool = False) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_command(job_name, eval_args)

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
    parser.add_argument("--env", required=True, choices=ENVS, help="Environment to evaluate")
    parser.add_argument("--method", required=True, choices=list(METHOD_CONFIGS.keys()), help="Evaluation method")
    parser.add_argument("--prefix", required=True, help="Experiment group prefix")
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

    # Get config file path
    config_file = METHOD_CONFIGS[args.method]
    config_path = f"configs/eval/{args.env}/{config_file}"

    # Loop over experiment IDs 0-4
    for exp_id in range(5):
        # Get checkpoints for this experiment
        checkpoints = get_checkpoints(args.env, exp_id)
        if args.sim:
            checkpoints["sim"] = args.sim
        if args.weak:
            checkpoints["weak"] = args.weak
        if args.strong:
            checkpoints["strong"] = args.strong

        # Validate checkpoints exist
        missing = False
        for name, path in checkpoints.items():
            if not Path(path).exists():
                print(f"Warning: exp{exp_id} {name} checkpoint not found: {path}")
                missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing checkpoints")
            continue

        # Build job name and experiment group
        method_name = METHOD_NAMES[args.method]
        job_name = f"{args.env}_{method_name}_exp{exp_id}"
        experiment_group = f"{args.prefix}_{args.env}_exp{exp_id}"

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
            **checkpoints,
        }

        submit_job(job_name, eval_args, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    exit(main())

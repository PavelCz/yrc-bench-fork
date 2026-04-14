#!/usr/bin/env python3
"""
Script to run gather_rollouts jobs in parallel via SLURM sbatch.
"""

import subprocess
from pathlib import Path

from common import get_checkpoints


# Conda environment
CONDA_ENV = "ood-stable"

# SLURM configuration
SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "mem": "256G",
    "cpus-per-task": "16",
}

# Default gather rollouts configuration
GATHER_DEFAULTS = {
    "wandb_mode": "offline",
    "config": "configs/procgen_gather.yaml",
    "random_percent": 0,
    "num_rollouts": 64,
    "use_bg": True,
    "query_cost": 0,
}

# Seed mapping for each experiment ID
EXP_ID_TO_SEED = {
    0: 6033,
    1: 1,
    2: 2,
}

# Server-specific paths
SERVER_PATHS = {
    "chai": {
        "checkpoint_base": "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        "seeds_base": "/nas/ucb/czempin/data/goal-misgen/seeds/icml",
    },
    "snoopy": {
        "checkpoint_base": "/scr/pavel/data/goal-misgen/policy/icml",
        "seeds_base": "/scr/pavel/data/goal-misgen/seeds/icml",
    },
}

# Environment choices
ENVS = ["coinrun", "maze"]


def build_sbatch_command(job_name: str, gather_args: dict) -> str:
    """Build the sbatch command string."""
    slurm_args = " ".join(f"--{k}={v}" for k, v in SLURM_CONFIG.items())

    python_args = [
        "python gather_rollouts.py",
        f"-wandb_mode {gather_args['wandb_mode']}",
        f"-c {gather_args['config']}",
        f"-n {gather_args['name']}",
        f"--experiment_group {gather_args['experiment_group']}",
        f"-en {gather_args['env_name']}",
        f"-random_percent {gather_args['random_percent']}",
        f"-sim {gather_args['sim']}",
        f"-weak {gather_args['weak']}",
        f"-strong {gather_args['strong']}",
        f"-num_rollouts={gather_args['num_rollouts']}",
        f"-use_bg={gather_args['use_bg']}",
        f"-seed {gather_args['seed']}",
        f"-level_seeds_file {gather_args['level_seeds_file']}",
        f"-query_cost {gather_args['query_cost']}",
    ]
    python_cmd = " ".join(python_args)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items())}

eval "$(conda shell.bash hook)"
conda activate {CONDA_ENV}
srun {slurm_args} {python_cmd}
"""
    return sbatch_script


def submit_job(job_name: str, gather_args: dict, dry_run: bool = False) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_command(job_name, gather_args)

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

    parser = argparse.ArgumentParser(description="Run gather_rollouts jobs via SLURM")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting"
    )
    parser.add_argument(
        "--server",
        choices=["chai", "snoopy"],
        default="snoopy",
        help="Server to use for paths (default: snoopy)",
    )
    parser.add_argument(
        "--env", required=True, choices=ENVS, help="Environment to gather rollouts for"
    )
    parser.add_argument("--prefix", required=True, help="Experiment group prefix")
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Experiment IDs to run (default: 0 1 2)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=GATHER_DEFAULTS["num_rollouts"],
        help="Number of rollouts",
    )
    parser.add_argument(
        "--random-percent",
        type=int,
        default=GATHER_DEFAULTS["random_percent"],
        help="Random percent for OOD",
    )
    parser.add_argument(
        "--config", default=GATHER_DEFAULTS["config"], help="Config file path"
    )
    parser.add_argument(
        "--wandb-mode", default=GATHER_DEFAULTS["wandb_mode"], help="Wandb mode"
    )
    parser.add_argument(
        "--use-bg", type=bool, default=GATHER_DEFAULTS["use_bg"], help="Use backgrounds"
    )
    parser.add_argument(
        "--query-cost",
        type=float,
        default=GATHER_DEFAULTS["query_cost"],
        help="Query cost",
    )
    # Override checkpoints if needed
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Get server-specific paths
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    seeds_base_path = paths["seeds_base"]

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {args.config}")
        print(f"Environment: {args.env}")
        print(f"Prefix: {args.prefix}")
        print(f"Experiment IDs: {args.exp_ids}")
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

        # Validate checkpoints and seeds file exist
        missing = False
        for name, path in checkpoints.items():
            if not Path(path).exists():
                print(f"Warning: exp{exp_id} {name} checkpoint not found: {path}")
                missing = True

        if not level_seeds_file.exists():
            print(
                f"Warning: exp{exp_id} level seeds file not found: {level_seeds_file}"
            )
            missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing files\n")
            continue

        # Build job name and experiment group
        job_name = f"gather_{args.env}_exp{exp_id}"
        experiment_group = f"{args.prefix}_{args.env}_exp{exp_id}"

        # Get seed for this experiment ID
        seed = EXP_ID_TO_SEED.get(exp_id, exp_id)

        if args.dry_run:
            print(f"=== exp{exp_id} ===")
            print(f"  Job name: {job_name}")
            print(f"  Experiment group: {experiment_group}")
            print(f"  Weak:   {checkpoints['weak']}")
            print(f"  Strong: {checkpoints['strong']}")
            print(f"  Level seeds: {level_seeds_file}")
            print(f"  Seed: {seed}")
            print()
            continue

        gather_args = {
            "config": args.config,
            "name": job_name,
            "experiment_group": experiment_group,
            "env_name": args.env,
            "random_percent": args.random_percent,
            "num_rollouts": args.num_rollouts,
            "use_bg": args.use_bg,
            "seed": seed,
            "query_cost": args.query_cost,
            "wandb_mode": args.wandb_mode,
            "level_seeds_file": str(level_seeds_file),
            **checkpoints,
        }

        submit_job(job_name, gather_args, dry_run=False)

    return 0


if __name__ == "__main__":
    exit(main())

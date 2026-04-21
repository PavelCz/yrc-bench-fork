#!/usr/bin/env python3
"""
Script to run DeepSVDD training jobs in parallel via SLURM sbatch.
"""

import subprocess
from pathlib import Path

from common import ENVS, SERVER_PATHS, get_checkpoints


# Conda environment
CONDA_ENV = "ood-stable"

# SLURM configuration
SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "mem": "256G",
}

# Default training configuration
TRAIN_DEFAULTS = {
    "config": "configs/procgen_ood.yaml",
    "cp_method": "DeepSVDD",
    "num_rollouts": 64,
    "query_cost": 0,
}

# Seed mapping for each experiment ID
EXP_ID_TO_SEED = {
    0: 6033,
    1: 1,
    2: 2,
}

# Feature type choices
FEATURE_TYPES = ["obs", "hidden"]


def get_rollout_dir(env: str, exp_id: int, rollouts_base_path: str) -> str:
    """Get the rollout directory path."""
    # Format: {rollouts_base}/{env}/gather_{env}_exp{id}/
    rollout_name = f"gather_{env}_exp{exp_id}"
    return str(Path(rollouts_base_path) / env / rollout_name)


def build_sbatch_command(job_name: str, train_args: dict) -> str:
    """Build the sbatch command string."""
    slurm_args = " ".join(f"--{k}={v}" for k, v in SLURM_CONFIG.items())

    python_args = [
        "python train_svdd.py",
        f"-wandb_group {train_args['wandb_group']}",
        f"-c {train_args['config']}",
        f"-n {train_args['name']}",
        f"-en {train_args['env_name']}",
        f"-sim {train_args['sim']}",
        f"-weak {train_args['weak']}",
        f"-strong {train_args['strong']}",
        f"-cp_method {train_args['cp_method']}",
        f"-cp_feature {train_args['feature_type']}",
        f"-rollout_dir {train_args['rollout_dir']}",
        f"-num_rollouts {train_args['num_rollouts']}",
        "-wandb",
        f"-query_cost {train_args['query_cost']}",
        f"-seed {train_args['seed']}",
        "-over",
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


def submit_job(job_name: str, train_args: dict, dry_run: bool = False) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_command(job_name, train_args)

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

    parser = argparse.ArgumentParser(description="Run DeepSVDD training jobs via SLURM")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting"
    )
    parser.add_argument(
        "--server",
        choices=list(SERVER_PATHS.keys()),
        default="snoopy",
        help="Server to use for paths (default: snoopy)",
    )
    parser.add_argument(
        "--env", required=True, choices=ENVS, help="Environment to train on"
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
        "--feature-type",
        required=True,
        choices=FEATURE_TYPES,
        help="Feature type: obs (observations) or hidden (latent)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=TRAIN_DEFAULTS["num_rollouts"],
        help="Number of rollouts",
    )
    parser.add_argument(
        "--config", default=TRAIN_DEFAULTS["config"], help="Config file path"
    )
    parser.add_argument(
        "--cp-method",
        default=TRAIN_DEFAULTS["cp_method"],
        help="Coordination policy method",
    )
    parser.add_argument(
        "--query-cost",
        type=float,
        default=TRAIN_DEFAULTS["query_cost"],
        help="Query cost",
    )
    parser.add_argument("--rollout-dir", help="Override rollout directory path")
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
    rollouts_base_path = paths["rollouts_base"]

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {args.config}")
        print(f"Environment: {args.env}")
        print(f"Feature type: {args.feature_type}")
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

        # Get rollout directory
        if args.rollout_dir:
            rollout_dir = args.rollout_dir
        else:
            rollout_dir = get_rollout_dir(args.env, exp_id, rollouts_base_path)

        # Get seed for this experiment ID
        seed = EXP_ID_TO_SEED.get(exp_id, exp_id)

        # Validate checkpoints exist
        missing = False
        for name, path in checkpoints.items():
            if not Path(path).exists():
                print(f"Warning: exp{exp_id} {name} checkpoint not found: {path}")
                missing = True

        # Check rollout directory exists
        if not Path(rollout_dir).exists():
            print(f"Warning: exp{exp_id} rollout directory not found: {rollout_dir}")
            missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing files\n")
            continue

        # Build job name and experiment group
        feature_suffix = "latent" if args.feature_type == "hidden" else "image"
        job_name = f"svdd_{args.env}_{feature_suffix}_exp{exp_id}"
        wandb_group = f"{args.prefix}_{args.env}_{feature_suffix}_exp{exp_id}"

        if args.dry_run:
            print(f"=== exp{exp_id} ===")
            print(f"  Job name: {job_name}")
            print(f"  Wandb group: {wandb_group}")
            print(f"  Feature type: {args.feature_type}")
            print(f"  Weak:   {checkpoints['weak']}")
            print(f"  Strong: {checkpoints['strong']}")
            print(f"  Rollout dir: {rollout_dir}")
            print(f"  Seed: {seed}")
            print()
            continue

        train_args = {
            "config": args.config,
            "name": job_name,
            "wandb_group": wandb_group,
            "env_name": args.env,
            "feature_type": args.feature_type,
            "cp_method": args.cp_method,
            "num_rollouts": args.num_rollouts,
            "query_cost": args.query_cost,
            "rollout_dir": rollout_dir,
            "seed": seed,
            **checkpoints,
        }

        submit_job(job_name, train_args, dry_run=False)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Script to run gather_rollouts jobs in parallel via SLURM sbatch.
"""

import subprocess
from pathlib import Path
from typing import Optional

from common import (
    ENVS,
    EXP_ID_TO_SEED,
    SERVER_PATHS,
    get_checkpoints,
    get_eval_env_name,
)


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
    "use_bg": True,
    "query_cost": 0,
    "rollout_chunk_size": None,
    "level_seeds_dir_name": "dummy_extra_ood_train_1024",
}


def parse_rollout_levels_arg(value: str):
    value = value.strip().lower()
    if value == "all":
        return None

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --num-levels value {value!r}. Use a positive integer or 'all'."
        ) from exc

    if parsed <= 0:
        raise ValueError(
            f"Invalid --num-levels value {parsed}. Use a positive integer or 'all'."
        )
    return parsed


def format_rollout_levels_label(num_levels) -> str:
    return "alllevels" if num_levels is None else f"{num_levels}levels"


def get_rollout_output_dir(rollouts_base_path: str, prefix: str, env: str) -> Path:
    return Path(rollouts_base_path) / prefix / env


def get_gather_env_name(env: str) -> str:
    """Map experiment env keys to the Procgen env used for rollout collection."""
    gather_env_name = get_eval_env_name(env)
    if gather_env_name == "maze":
        raise ValueError("Plain Procgen env 'maze' is not valid; use 'maze_afh'.")
    return gather_env_name


def get_default_level_seeds_dir(seeds_base_path: str) -> Path:
    return Path(seeds_base_path).parent / GATHER_DEFAULTS["level_seeds_dir_name"]


def get_level_seeds_file(
    exp_id: int, seeds_base_path: str, level_seeds_file: Optional[Path] = None
) -> Path:
    if level_seeds_file is not None:
        return level_seeds_file
    return Path(seeds_base_path) / f"{exp_id}.json"


def build_sbatch_command(job_name: str, gather_args: dict) -> str:
    """Build the sbatch command string."""
    if gather_args["env_name"] == "maze":
        raise ValueError("Plain Procgen env 'maze' is not valid; use 'maze_afh'.")

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
        f"-use_bg={gather_args['use_bg']}",
        f"-seed {gather_args['seed']}",
        f"-level_seeds_file {gather_args['level_seeds_file']}",
        f"-query_cost {gather_args['query_cost']}",
    ]
    if gather_args["rollout_levels"] is not None:
        python_args.append(f"-rollout_levels {gather_args['rollout_levels']}")
    if gather_args["rollout_chunk_size"] is not None:
        python_args.append(f"-rollout_chunk_size {gather_args['rollout_chunk_size']}")
    python_cmd = " ".join(python_args)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items())}

eval "$(conda shell.bash hook)"
conda activate {CONDA_ENV}
export SM_OUTPUT_DIR="{gather_args["output_dir"]}"
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
        choices=["cluster1", "cluster2"],
        default="cluster2",
        help="Server to use for paths (default: cluster2)",
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
        "--num-levels",
        type=parse_rollout_levels_arg,
        nargs="+",
        default=None,
        help=(
            "Number(s) of ood_train levels to gather. Use positive integers and/or "
            "'all'. Omit the flag to use all available ood_train seeds."
        ),
    )
    parser.add_argument(
        "--level-seeds-file",
        type=Path,
        default=None,
        help=(
            "Override the level seed JSON file. This is useful for alternate "
            "OOD-train-only seed files and requires exactly one --exp-ids value."
        ),
    )
    parser.add_argument(
        "--level-seeds-dir",
        type=Path,
        default=None,
        help=(
            "Override the directory containing {exp_id}.json level seed files. "
            "Defaults to the selected server's dummy_extra_ood_train_1024 "
            "seed directory."
        ),
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
    parser.add_argument(
        "--rollout-chunk-size",
        type=int,
        default=GATHER_DEFAULTS["rollout_chunk_size"],
        help=(
            "Maximum observations per rollout chunk in gather_rollouts.py. "
            "Omit to use gather_rollouts.py's default chunk size; use 0 to "
            "disable chunked saving."
        ),
    )
    # Override checkpoints if needed
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    args = parser.parse_args()

    if args.level_seeds_file is not None and args.level_seeds_dir is not None:
        parser.error("Pass either --level-seeds-file or --level-seeds-dir, not both.")
    if args.level_seeds_file is not None and len(args.exp_ids) != 1:
        parser.error("--level-seeds-file requires exactly one --exp-ids value.")

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Get server-specific paths
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    rollouts_base_path = paths["rollouts_base"]
    seeds_base_path = str(
        args.level_seeds_dir or get_default_level_seeds_dir(paths["seeds_base"])
    )
    rollout_output_dir = get_rollout_output_dir(
        rollouts_base_path, args.prefix, args.env
    )
    gather_env_name = get_gather_env_name(args.env)

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {args.config}")
        print(f"Environment: {args.env}")
        print(f"Procgen env name: {gather_env_name}")
        print(f"Prefix: {args.prefix}")
        print(f"Rollout output dir: {rollout_output_dir}")
        if args.level_seeds_file is not None:
            print(f"Level seeds file override: {args.level_seeds_file}")
        else:
            print(f"Level seeds dir: {seeds_base_path}")
        print(f"Experiment IDs: {args.exp_ids}")
        rollout_level_counts = [
            "all" if num_levels is None else num_levels
            for num_levels in (
                args.num_levels if args.num_levels is not None else [None]
            )
        ]
        print(f"Rollout level counts: {rollout_level_counts}")
        print(f"Rollout chunk size: {args.rollout_chunk_size}")
        print()

    requested_rollout_levels = (
        args.num_levels if args.num_levels is not None else [None]
    )

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
        level_seeds_file = get_level_seeds_file(
            exp_id, seeds_base_path, args.level_seeds_file
        )

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

        # Build run name and experiment group
        run_name = f"gather_{args.env}_exp{exp_id}"
        experiment_group = f"{args.prefix}_{args.env}_exp{exp_id}"

        # Get seed for this experiment ID
        seed = EXP_ID_TO_SEED.get(exp_id, exp_id)

        for rollout_levels in requested_rollout_levels:
            rollout_levels_label = format_rollout_levels_label(rollout_levels)
            job_name = (
                run_name
                if rollout_levels is None and len(requested_rollout_levels) == 1
                else f"{run_name}_{rollout_levels_label}"
            )

            gather_args = {
                "config": args.config,
                "name": run_name,
                "experiment_group": experiment_group,
                "env_name": gather_env_name,
                "random_percent": args.random_percent,
                "rollout_levels": rollout_levels,
                "use_bg": args.use_bg,
                "seed": seed,
                "query_cost": args.query_cost,
                "rollout_chunk_size": args.rollout_chunk_size,
                "wandb_mode": args.wandb_mode,
                "output_dir": str(rollout_output_dir),
                "level_seeds_file": str(level_seeds_file),
                **checkpoints,
            }

            if args.dry_run:
                print(f"=== exp{exp_id} / {rollout_levels_label} ===")
                print(f"  Job name: {job_name}")
                print(f"  Run name: {run_name}")
                print(f"  Experiment group: {experiment_group}")
                print(f"  Output dir: {rollout_output_dir}")
                print(f"  Expected rollout dir: {rollout_output_dir / run_name}")
                print(f"  Procgen env name: {gather_env_name}")
                print(f"  Weak:   {checkpoints['weak']}")
                print(f"  Strong: {checkpoints['strong']}")
                print(f"  Level seeds: {level_seeds_file}")
                print(f"  Rollout levels: {rollout_levels_label}")
                print(f"  Seed: {seed}")
                print()

            submit_job(job_name, gather_args, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    exit(main())

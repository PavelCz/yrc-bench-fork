#!/usr/bin/env python3
"""
Script to run eval_strong_on_help.py jobs in parallel via SLURM sbatch.

This script finds existing evaluation NPZ files and re-evaluates 
the strong agent on the seeds where help was requested.
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


# Default conda environment
DEFAULT_CONDA_ENV = "ood-stable"

# SLURM configuration (lighter weight than full eval)
SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",  # Shorter time since we're only re-evaluating
    "mem": "50G",
    "cpus-per-task": "16",
}

# Server-specific paths
SERVER_PATHS = {
    "chai": {
        "checkpoint_base": "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        "evals_base": "/nas/ucb/czempin/data/goal-misgen/experiments/evals",
    },
    "snoopy": {
        "checkpoint_base": "/scr/pavel/data/goal-misgen/policy/icml",
        "evals_base": "/scr/pavel/data/goal-misgen/experiments/evals",
    },
}

# Environment choices
ENVS = ["maze", "coinrun"]

# Method to config file mapping (same as run_eval.py)
METHOD_CONFIGS = {
    "max_prob": "max_prob.yaml",
    "max_logit": "max_logit.yaml",
    "lb_random": "level_based_random.yaml",
    "ts_random": "timestep_random.yaml",
    "svdd_image": "image_svdd.yaml",
    "svdd_latent": "latent_svdd.yaml",
    "ensemble": "ensemble_variance.yaml",
    "ensemble_single": "ensemble_variance_single.yaml",
    "wait": "wait.yaml",
}

EXPECTED_TIMESTEPS = 200015872


def get_env_folder(env: str) -> str:
    """Get the environment folder name."""
    if env == "coinrun":
        return "coinrun"
    else:
        return f"{env}_afh"


def find_newest_timestamp_dir(parent_dir: Path) -> Optional[Path]:
    """Find the newest timestamp directory in parent_dir.

    Looks for dirs matching format: YYYY-MM-DD__HH-MM-SS__seed_* or YYYYMMDD_HHMMSS
    Returns the newest one based on the timestamp in the name.
    """
    if not parent_dir.exists():
        return None

    # Find all timestamp directories
    timestamp_dirs = []
    for d in parent_dir.iterdir():
        if d.is_dir():
            # Match either format: __seed_ or pure timestamp YYYYMMDD_HHMMSS
            if "__seed_" in d.name or re.match(r"^\d{8}_\d{6}$", d.name):
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
        print(
            f"Warning: {ts_dir.name} has max timesteps {highest_timesteps}, expected {EXPECTED_TIMESTEPS}"
        )

    return best_model


def get_strong_checkpoint(env: str, exp_id: int, checkpoint_base_path: str) -> Optional[str]:
    """Get strong checkpoint path based on environment and experiment ID."""
    env_folder = get_env_folder(env)
    base_path = Path(checkpoint_base_path) / env_folder

    strong_parent = base_path / f"icml2_{env}_exp{exp_id}_50p"
    strong_ts_dir = find_newest_timestamp_dir(strong_parent)
    strong_model = find_best_model_checkpoint(strong_ts_dir) if strong_ts_dir else None

    return str(strong_model) if strong_model else None


def find_eval_npz_files(
    evals_base: str, 
    prefix: str, 
    env: str, 
    exp_ids: List[int],
    method_filter: Optional[str] = None,
) -> List[Tuple[Path, str, int, str]]:
    """Find NPZ files from existing evaluations.
    
    Args:
        evals_base: Base path for evaluations
        prefix: Experiment group prefix to match
        env: Environment name
        exp_ids: List of experiment IDs to search
        method_filter: Optional method name filter (e.g., "max_prob")
    
    Returns:
        List of tuples: (npz_path, method_name, exp_id)
    """
    evals_path = Path(evals_base)
    results = []
    
    for exp_id in exp_ids:
        # Look for experiment group directories matching pattern: {prefix}_{env}_exp{exp_id}
        exp_group_pattern = f"{prefix}_{env}_exp{exp_id}"
        exp_group_dir = evals_path / exp_group_pattern
        
        if not exp_group_dir.exists():
            print(f"Warning: Experiment group dir not found: {exp_group_dir}")
            continue
            
        # Look for method subdirectories: {env}_{method}_exp{exp_id}
        for method_dir in exp_group_dir.iterdir():
            if not method_dir.is_dir():
                continue
                
            # Extract method name from directory name
            # Format: {env}_{method}_exp{exp_id}
            match = re.match(rf"{env}_(.+)_exp{exp_id}$", method_dir.name)
            if not match:
                continue
                
            method_name = match.group(1)
            
            # Apply method filter if specified
            if method_filter and method_name != method_filter:
                continue
            
            # Find the timestamp directory
            ts_dir = find_newest_timestamp_dir(method_dir)
            if not ts_dir:
                print(f"Warning: No timestamp dir found in {method_dir}")
                continue
            
            # Look for NPZ files (e.g., eval_seed_1_test.npz)
            npz_files = list(ts_dir.glob("eval_seed_*_test.npz"))
            if not npz_files:
                print(f"Warning: No NPZ files found in {ts_dir}")
                continue
            
            for npz_file in npz_files:
                results.append((npz_file, method_name, exp_id))
    
    return results


def build_sbatch_command(
    job_name: str, 
    npz_file: str, 
    strong_path: str, 
    config_path: str,
    conda_env: str, 
    qos: str = "default"
) -> str:
    """Build the sbatch command string."""
    # Override QOS in SLURM config
    slurm_config = SLURM_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())

    # Build the python command
    python_cmd = f"python eval_strong_on_help.py -c {config_path} --npz_file {npz_file} -strong {strong_path}"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""
    return sbatch_script


def submit_job(
    job_name: str,
    npz_file: str,
    strong_path: str,
    config_path: str,
    conda_env: str,
    qos: str = "default",
    dry_run: bool = False,
) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_command(job_name, npz_file, strong_path, config_path, conda_env, qos)

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
    parser = argparse.ArgumentParser(
        description="Run eval_strong_on_help.py jobs via SLURM to re-evaluate strong agent on help-requested seeds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting"
    )
    parser.add_argument(
        "--conda-env",
        default=DEFAULT_CONDA_ENV,
        help=f"Conda environment to use (default: {DEFAULT_CONDA_ENV})",
    )
    parser.add_argument(
        "--server",
        choices=["chai", "snoopy"],
        default="chai",
        help="Server to use for paths (default: chai)",
    )
    parser.add_argument(
        "--qos",
        choices=["default", "high"],
        default="default",
        help="SLURM QOS to use (default: default)",
    )
    parser.add_argument(
        "--env", required=True, choices=ENVS, help="Environment to evaluate"
    )
    parser.add_argument(
        "--prefix", required=True, help="Experiment group prefix to match"
    )
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Experiment IDs to run (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Filter to specific method (e.g., max_prob). If not set, processes all methods found.",
    )
    parser.add_argument(
        "--strong",
        default=None,
        help="Override strong checkpoint path (applies to all jobs)",
    )
    parser.add_argument(
        "--evals-base",
        default=None,
        help="Override evals base path",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override config file path (YAML config, e.g., configs/eval/coinrun/max_prob.yaml)",
    )
    args = parser.parse_args()

    # Get server-specific paths
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    evals_base = args.evals_base or paths["evals_base"]

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Evals base: {evals_base}")
        print(f"Environment: {args.env}")
        print(f"Prefix: {args.prefix}")
        print(f"Method filter: {args.method or 'all'}")
        print()

    # Find all NPZ files matching criteria
    npz_files = find_eval_npz_files(
        evals_base, args.prefix, args.env, args.exp_ids, args.method
    )

    if not npz_files:
        print("No NPZ files found matching criteria")
        return 1

    print(f"Found {len(npz_files)} NPZ files to process")
    if args.dry_run:
        print()

    # Process each NPZ file
    submitted = 0
    skipped = 0
    
    for npz_path, method_name, exp_id in npz_files:
        # Get strong checkpoint
        if args.strong:
            strong_path = args.strong
        else:
            strong_path = get_strong_checkpoint(args.env, exp_id, checkpoint_base_path)
        
        if not strong_path:
            print(f"Warning: Strong checkpoint not found for exp{exp_id}, skipping")
            skipped += 1
            continue
        
        if not Path(strong_path).exists():
            print(f"Warning: Strong checkpoint does not exist: {strong_path}, skipping")
            skipped += 1
            continue
        
        # Determine config path - use YAML config based on method name
        if args.config:
            effective_config = args.config
        elif method_name in METHOD_CONFIGS:
            effective_config = f"configs/eval/{args.env}/{METHOD_CONFIGS[method_name]}"
        else:
            print(f"Warning: Unknown method '{method_name}' for {npz_path}, skipping")
            skipped += 1
            continue
        
        if not Path(effective_config).exists():
            print(f"Warning: Config not found: {effective_config}, skipping")
            skipped += 1
            continue
        
        # Check if output already exists
        output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
        if output_path.exists():
            print(f"Skipping {npz_path.name} - output already exists: {output_path}")
            skipped += 1
            continue
        
        # Build job name
        job_name = f"strong_reval_{args.env}_{method_name}_exp{exp_id}"
        
        if args.dry_run:
            print(f"=== {job_name} ===")
            print(f"  NPZ file: {npz_path}")
            print(f"  Strong:   {strong_path}")
            print(f"  Config:   {effective_config}")
            print(f"  Output:   {output_path}")
            print()
        else:
            submit_job(
                job_name,
                str(npz_path),
                strong_path,
                effective_config,
                args.conda_env,
                args.qos,
                dry_run=False,
            )
        
        submitted += 1

    print(f"\nSummary: {submitted} jobs {'would be ' if args.dry_run else ''}submitted, {skipped} skipped")
    return 0


if __name__ == "__main__":
    exit(main())

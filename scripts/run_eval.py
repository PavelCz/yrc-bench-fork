#!/usr/bin/env python3
"""
Script to run evaluation jobs in parallel via SLURM sbatch.
"""

import subprocess
from datetime import date
from pathlib import Path
from typing import List, Optional

from common import (
    ENSEMBLE_METHODS,
    ENVS,
    METHOD_CONFIGS,
    SERVER_PATHS,
    SVDD_METHODS,
    find_best_model_checkpoint,
    find_newest_timestamp_dir,
    get_checkpoints,
    get_env_folder,
)


# Default conda environment
DEFAULT_CONDA_ENV = "ood-stable"
REPO_ROOT = Path(__file__).resolve().parents[1]

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
    "video_logging_mode": "folder",
    "video_filter_mode": "any",
    "coverage_fraction": 0.05,
}

# Default number of ensemble members (excluding weak agent which is added automatically)
DEFAULT_NUM_ENSEMBLE_MEMBERS = 4

EVAL_ENVS = [*ENVS, "coinrun_proxy_fail", "maze_proxy_fail"]

# Some evaluation environments intentionally reuse training artifacts from a
# base environment. `coinrun_proxy_fail` and `maze_proxy_fail` change the
# Procgen reward/termination behavior at evaluation time, but their
# weak/strong checkpoints, SVDD models, ensemble members, and YAML configs are
# still the normal coinrun/maze artifacts. Keep `args.env` for the environment
# passed to eval_afhp.py; use this alias only for filesystem lookup paths.
ARTIFACT_ENVS = {
    "coinrun_proxy_fail": "coinrun",
    "maze_proxy_fail": "maze",
}


def get_svdd_feature_type(method: str) -> str:
    """Get the SVDD feature type from method name."""
    if method == "svdd-image":
        return "image"
    elif method == "svdd-latent":
        return "latent"
    return ""


def get_svdd_policy_name(env: str, exp_id: int, method: str) -> str:
    """Get the SVDD policy directory name."""
    feature_type = get_svdd_feature_type(method)
    # Format: svdd_{env}_{feature_type}_exp{id}
    return f"svdd_{env}_{feature_type}_exp{exp_id}"


def get_svdd_model_path(
    env: str,
    exp_id: int,
    method: str,
    svdd_base_path: str,
    svdd_prefix: Optional[str] = None,
) -> Optional[str]:
    """Get the full path to the trained SVDD model file, or None if it doesn't exist."""
    policy_name = get_svdd_policy_name(env, exp_id, method)
    model_base_path = Path(svdd_base_path)
    if svdd_prefix is not None:
        model_base_path = model_base_path / svdd_prefix
    model_file = model_base_path / policy_name / "trained.joblib"
    if model_file.exists():
        return str(model_file)
    return None


def get_svdd_expected_model_path(
    env: str,
    exp_id: int,
    method: str,
    svdd_base_path: str,
    svdd_prefix: Optional[str] = None,
) -> Path:
    policy_name = get_svdd_policy_name(env, exp_id, method)
    model_base_path = Path(svdd_base_path)
    if svdd_prefix is not None:
        model_base_path = model_base_path / svdd_prefix
    return model_base_path / policy_name / "trained.joblib"


def get_ensemble_member_paths(
    env: str,
    exp_id: int,
    checkpoint_base_path: str,
    num_members: int = DEFAULT_NUM_ENSEMBLE_MEMBERS,
) -> List[Optional[str]]:
    """Get paths to ensemble member checkpoints.

    Ensemble members are stored at:
    {checkpoint_base}/ensembles/icml2_ensemble_{env}_exp{id}_m{member_id}/...

    Args:
        env: Environment name (maze, coinrun)
        exp_id: Experiment ID
        checkpoint_base_path: Base path for checkpoints
        num_members: Number of ensemble members to find

    Returns:
        List of paths to ensemble member checkpoints, or None for missing members
    """
    env_folder = get_env_folder(env)
    base_path = Path(checkpoint_base_path) / env_folder / "ensembles"

    member_paths = []
    for member_id in range(num_members):
        member_parent = base_path / f"icml2_ensemble_{env}_exp{exp_id}_m{member_id}"
        member_ts_dir = find_newest_timestamp_dir(member_parent)
        member_model = (
            find_best_model_checkpoint(member_ts_dir) if member_ts_dir else None
        )

        if member_model:
            member_paths.append(str(member_model))
        else:
            member_paths.append(None)

    return member_paths


def build_sbatch_command(
    job_name: str, eval_args: dict, conda_env: str, log_dir: Path, qos: str = "default"
) -> str:
    """Build the sbatch command string."""
    # Override QOS in SLURM config
    slurm_config = SLURM_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())

    # Build the python command arguments
    python_args = [
        "python eval_afhp.py",
        f"-c {eval_args['config']}",
        f"-n {eval_args['name']}",
        f"-en {eval_args['env_name']}",
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
        f"-coverage_fraction {eval_args['coverage_fraction']}",
    ]

    # Add wandb project if specified
    if eval_args.get("wandb_project"):
        python_args.append(f"-wandb_project {eval_args['wandb_project']}")

    # Add SVDD-specific arguments if present
    if eval_args.get("cp_feature"):
        python_args.append(f"-cp_feature {eval_args['cp_feature']}")
    if eval_args.get("svdd_model_path"):
        python_args.append(f"-f_n {eval_args['svdd_model_path']}")

    # Add ensemble-specific arguments if present
    if eval_args.get("ensemble_members"):
        python_args.append("-cp_ensemble_members")
        for member_path in eval_args["ensemble_members"]:
            python_args.append(f"    {member_path}")

    # Join python args as single line
    python_cmd = " ".join(python_args)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

echo "Using conda env: {conda_env}"
eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""
    return sbatch_script


def submit_job(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    qos: str = "default",
    dry_run: bool = False,
) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_command(job_name, eval_args, conda_env, log_dir, qos)

    if dry_run:
        print(f"=== Job: {job_name} ===")
        print(sbatch_script)
        print()
        return

    # Ensure logs directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

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


def run_preflight_check(conda_env: str, env_name: str, *, show_output: bool) -> bool:
    """Run local dependency checks before submitting SLURM jobs."""
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "scripts.preflight_eval_env",
        "--env",
        env_name,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    if show_output or result.returncode != 0:
        print("=== Preflight check ===")
        print("$ " + " ".join(command))
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    if result.returncode != 0:
        print(
            f"Preflight failed with exit code {result.returncode}; "
            "not submitting jobs."
        )
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation jobs via SLURM")
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
        "--env", required=True, choices=EVAL_ENVS, help="Environment to evaluate"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHOD_CONFIGS.keys()),
        help="Evaluation method",
    )
    parser.add_argument("--prefix", required=True, help="Experiment group prefix")
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Experiment IDs to run (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=EVAL_DEFAULTS["num_levels"],
        help="Number of levels",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=EVAL_DEFAULTS["video_episodes_to_collect"],
        help="Video episodes to collect",
    )
    parser.add_argument(
        "--video-filter", default=EVAL_DEFAULTS["video_filter"], help="Video filter"
    )
    parser.add_argument(
        "--cp-rolling-average",
        default=EVAL_DEFAULTS["cp_rolling_average"],
        help="Coordination policy rolling average",
    )
    parser.add_argument(
        "--video-logging-mode",
        default=EVAL_DEFAULTS["video_logging_mode"],
        help="Video logging mode",
    )
    parser.add_argument(
        "--video-filter-mode",
        default=EVAL_DEFAULTS["video_filter_mode"],
        help="Video filter mode",
    )
    parser.add_argument(
        "--coverage-fraction",
        type=float,
        default=EVAL_DEFAULTS["coverage_fraction"],
        help=f"Coverage fraction for threshold sampling (default: {EVAL_DEFAULTS['coverage_fraction']})",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override wandb project name",
    )
    parser.add_argument(
        "--svdd-prefix",
        default=None,
        help=(
            "SVDD training prefix under the server svdd_base path. Defaults to "
            "--prefix for SVDD methods."
        ),
    )
    # Override checkpoints if needed
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    # Ensemble-specific arguments
    parser.add_argument(
        "--num-ensemble-members",
        type=int,
        default=DEFAULT_NUM_ENSEMBLE_MEMBERS,
        help=f"Number of ensemble members (default: {DEFAULT_NUM_ENSEMBLE_MEMBERS})",
    )
    args = parser.parse_args()

    # Get server-specific paths
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    seeds_base_path = paths["seeds_base"]
    svdd_base_path = paths["svdd_base"]
    svdd_prefix = args.svdd_prefix or args.prefix

    # `args.env` is the actual Procgen environment to instantiate. `artifact_env`
    # is only the namespace used for existing experiment artifacts on disk.
    artifact_env = ARTIFACT_ENVS.get(args.env, args.env)

    # Configs are artifact-scoped: proxy-fail evaluation reuses coinrun configs
    # and overrides the actual env through -en below.
    config_file = METHOD_CONFIGS[args.method]
    config_path = f"configs/eval/{artifact_env}/{config_file}"

    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Conda env: {args.conda_env}")

    # Build log directory path: base / wandb_project / prefix / date
    log_base = Path(paths["log_base"])
    wandb_project = args.wandb_project or "default"
    log_dir = log_base / wandb_project / args.prefix / date.today().isoformat()
    print(f"Log dir: {log_dir}")

    if not run_preflight_check(
        args.conda_env,
        args.env,
        show_output=args.dry_run,
    ):
        return 1

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {config_path}")
        print(f"Environment: {args.env}")
        if artifact_env != args.env:
            print(f"Artifact env: {artifact_env}")
        print(f"Method: {args.method}")
        print(f"Prefix: {args.prefix}")
        if args.method in SVDD_METHODS:
            print(f"SVDD prefix: {svdd_prefix}")
            print(f"SVDD base: {Path(svdd_base_path) / svdd_prefix}")
        print()

    # Loop over experiment IDs
    for exp_id in args.exp_ids:
        # Checkpoints and method-specific models are artifact-scoped. Job names
        # and experiment groups below stay env-scoped so outputs are labeled by
        # the actual evaluation environment.
        checkpoints = get_checkpoints(artifact_env, exp_id, checkpoint_base_path)
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
            svdd_model_path = get_svdd_model_path(
                artifact_env, exp_id, args.method, svdd_base_path, svdd_prefix
            )
            cp_feature = "obs" if args.method == "svdd-image" else "hidden"

        # Get ensemble-specific settings if needed
        ensemble_members = None
        if args.method in ENSEMBLE_METHODS:
            ensemble_members = get_ensemble_member_paths(
                artifact_env, exp_id, checkpoint_base_path, args.num_ensemble_members
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

        if args.method in SVDD_METHODS and svdd_model_path is None:
            expected_model_path = get_svdd_expected_model_path(
                artifact_env, exp_id, args.method, svdd_base_path, svdd_prefix
            )
            print(
                f"Warning: exp{exp_id} SVDD model not found at {expected_model_path}"
            )
            missing = True

        if args.method in ENSEMBLE_METHODS and ensemble_members:
            for i, member_path in enumerate(ensemble_members):
                if member_path is None:
                    env_folder = get_env_folder(artifact_env)
                    print(
                        f"Warning: exp{exp_id} ensemble member m{i} not found at "
                        f"{checkpoint_base_path}/{env_folder}/ensembles/icml2_ensemble_{artifact_env}_exp{exp_id}_m{i}/"
                    )
                    missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing files\n")
            continue

        # Build job name and experiment group
        job_name = f"{args.env}_{args.method}_exp{exp_id}"
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
            if ensemble_members:
                print(f"  Ensemble members ({len(ensemble_members)}):")
                for i, member_path in enumerate(ensemble_members):
                    print(f"    m{i}: {member_path}")
            print()
            continue

        eval_args = {
            "config": config_path,
            "name": job_name,
            # build_sbatch_command emits this as `-en`, so eval_afhp.py creates
            # the requested environment even when config/artifact lookup used an
            # alias such as coinrun.
            "env_name": args.env,
            "experiment_group": experiment_group,
            "num_levels": args.num_levels,
            "video_episodes_to_collect": args.video_episodes,
            "video_filter": args.video_filter,
            "cp_rolling_average": args.cp_rolling_average,
            "video_logging_mode": args.video_logging_mode,
            "video_filter_mode": args.video_filter_mode,
            "coverage_fraction": args.coverage_fraction,
            "wandb_project": args.wandb_project,
            "level_seeds_file": str(level_seeds_file),
            "svdd_model_path": svdd_model_path,
            "cp_feature": cp_feature,
            "ensemble_members": ensemble_members,
            **checkpoints,
        }

        submit_job(
            job_name, eval_args, args.conda_env, log_dir, args.qos, dry_run=False
        )

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Script to run evaluation jobs in parallel via SLURM sbatch.
"""

import re
import subprocess
from datetime import date
from pathlib import Path
from typing import List, Optional


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
    "video_logging_mode": "folder",
    "video_filter_mode": "any",
    "num_bins": 20,
}

# Server-specific paths
SERVER_PATHS = {
    "chai": {
        "checkpoint_base": "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        "seeds_base": "/nas/ucb/czempin/data/goal-misgen/seeds/icml",
        "svdd_base": "/nas/ucb/czempin/data/goal-misgen/trained_svdd",
        "log_base": "/nas/ucb/czempin/data/goal-misgen/slurm-logs",
    },
    "snoopy": {
        "checkpoint_base": "/scr/pavel/data/goal-misgen/policy/icml",
        "seeds_base": "/scr/pavel/data/goal-misgen/seeds/icml",
        "svdd_base": "/scr/pavel/data/goal-misgen/trained_svdd",
        "log_base": "/scr/pavel/data/goal-misgen/slurm-logs",
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
    "ensemble": "ensemble_variance.yaml",
    "ensemble-single": "ensemble_variance_single.yaml",
    "wait": "wait.yaml",
}

# Method to run name suffix mapping
METHOD_NAMES = {
    "max-prob": "max_prob",
    "max-logit": "max_logit",
    "lb-random": "lb_random",
    "ts-random": "ts_random",
    "svdd-image": "svdd_image",
    "svdd-latent": "svdd_latent",
    "ensemble": "ensemble",
    "ensemble-single": "ensemble_single",
    "wait": "wait",
}

# Methods that require a trained SVDD policy
SVDD_METHODS = {"svdd-image", "svdd-latent"}

# Methods that require ensemble members
ENSEMBLE_METHODS = {"ensemble", "ensemble-single"}

# Default number of ensemble members (excluding weak agent which is added automatically)
DEFAULT_NUM_ENSEMBLE_MEMBERS = 4

CHECKPOINT_TIMESTEP_PATTERN = re.compile(r"model_(\d+)\.(?:pth|ckpt)$")
THOUSAND = 1_000
MILLION = 1_000_000


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
        print(
            f"Warning: {ts_dir.name} has max timesteps {highest_timesteps}, expected {EXPECTED_TIMESTEPS}"
        )

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


def get_svdd_model_path(
    env: str, exp_id: int, method: str, svdd_base_path: str
) -> Optional[str]:
    """Get the full path to the trained SVDD model file, or None if it doesn't exist."""
    policy_name = get_svdd_policy_name(env, exp_id, method)
    model_file = Path(svdd_base_path) / policy_name / "trained.joblib"
    if model_file.exists():
        return str(model_file)
    return None


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


def _extract_timestep_from_path(path: Optional[str]) -> Optional[int]:
    """Extract timestep from checkpoint-like filenames (e.g. model_200000.pth)."""
    if not path:
        return None

    name = Path(path).name
    match = CHECKPOINT_TIMESTEP_PATTERN.search(name)
    if match:
        return int(match.group(1))
    return None


def _format_timestep_suffix(timestep: Optional[int]) -> Optional[str]:
    """Format timestep into a short readable suffix (e.g. 200k, 10M)."""
    if timestep is None:
        return None
    if timestep >= MILLION:
        value = timestep / MILLION
        unit = "M"
    elif timestep >= THOUSAND:
        value = timestep / THOUSAND
        unit = "k"
    else:
        return str(timestep)

    if float(value).is_integer():
        formatted = str(int(value))
    else:
        formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}{unit}"


def _calibration_timestep_suffix(eval_args: dict) -> Optional[str]:
    """Pick a representative timestep suffix from relevant policy checkpoints.

    Priority order:
    1) svdd_model_path (coordination policy checkpoint, when method uses it),
    2) weak policy checkpoint (primary acting policy reference),
    3) sim policy checkpoint,
    4) strong policy checkpoint.
    """
    for path in (
        eval_args.get("svdd_model_path"),
        eval_args.get("weak"),
        eval_args.get("sim"),
        eval_args.get("strong"),
    ):
        timestep = _extract_timestep_from_path(path)
        if timestep is not None:
            return _format_timestep_suffix(timestep)
    return None


def resolve_calibration_path(job_name: str, eval_args: dict) -> Path:
    """Resolve the pre-computed calibration artifact path for an eval job."""
    suffix = _calibration_timestep_suffix(eval_args)
    suffix_part = f"_{suffix}" if suffix is not None else ""
    filename = f"{job_name}_calibration{suffix_part}.npz"

    weak_ckpt = eval_args.get("weak")
    if weak_ckpt:
        return Path(weak_ckpt).parent / filename

    return Path(filename)


def _ensure_calibration_exists(job_name: str, calibration_path: Path) -> bool:
    if calibration_path.exists():
        return True
    print(
        f"  Missing calibration file for {job_name}: {calibration_path}. "
        "Run calibrate_afhp.py with matching checkpoints first, then retry."
    )
    return False


def _build_policy_python_args(script: str, eval_args: dict) -> List[str]:
    """Build the python args shared across all job types (policy loading, env setup)."""
    args = [
        f"python {script}",
        f"-c {eval_args['config']}",
        f"-n {eval_args['name']}",
        "-defer_to_oracle",
        f"-experiment_group {eval_args['experiment_group']}",
        f"-num_levels={eval_args['num_levels']}",
        f"-sim {eval_args['sim']}",
        f"-weak {eval_args['weak']}",
        f"-strong {eval_args['strong']}",
        f"-level_seeds_file {eval_args['level_seeds_file']}",
    ]
    if eval_args.get("wandb_project"):
        args.append(f"-wandb_project {eval_args['wandb_project']}")
    if eval_args.get("cp_feature"):
        args.append(f"-cp_feature {eval_args['cp_feature']}")
    if eval_args.get("svdd_model_path"):
        args.append(f"-f_n {eval_args['svdd_model_path']}")
    if eval_args.get("ensemble_members"):
        args.append("-cp_ensemble_members")
        for member_path in eval_args["ensemble_members"]:
            args.append(f"    {member_path}")
    return args


def build_sequential_eval_sbatch_command(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    calibration_path: Path,
    qos: str = "default",
) -> str:
    """Build the sbatch script for a sequential AFHP evaluation job."""
    slurm_config = SLURM_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())

    python_args = _build_policy_python_args("eval_afhp_seq.py", eval_args)
    python_args += [
        f"--calibration_path {calibration_path}",
        f"-video_episodes_to_collect {eval_args['video_episodes_to_collect']}",
        f"-video_filter {eval_args['video_filter']}",
        f"-cp_rolling_average {eval_args['cp_rolling_average']}",
        f"-video_logging_mode={eval_args['video_logging_mode']}",
        f"-video_filter_mode={eval_args['video_filter_mode']}",
        f"-num_bins {eval_args['num_bins']}",
    ]
    python_cmd = " ".join(python_args)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_seq
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

echo "Using conda env: {conda_env}"
eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""
    return sbatch_script


def build_bin_array_sbatch_command(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    calibration_path: Path,
    num_bins: int,
    qos: str = "default",
) -> str:
    """Build the sbatch script for the bin evaluation array job."""
    slurm_config = SLURM_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())

    # $SLURM_ARRAY_TASK_ID must expand at runtime, not be interpolated by Python
    checkpoint_path = f"{log_dir}/{job_name}_bin_$SLURM_ARRAY_TASK_ID.npz"
    python_args = _build_policy_python_args("eval_afhp_bin.py", eval_args)
    python_args += [
        "--bin_idx $SLURM_ARRAY_TASK_ID",
        f"--checkpoint_path {checkpoint_path}",
        f"--calibration_path {calibration_path}",
    ]
    python_cmd = " ".join(python_args)

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}_bin
#SBATCH --output={log_dir}/%x_%j_%a.out
#SBATCH --error={log_dir}/%x_%j_%a.err
#SBATCH --array=0-{num_bins - 1}
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

echo "Bin $SLURM_ARRAY_TASK_ID / {num_bins} for {job_name}"
eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""


def _sbatch_submit(script: str, log_dir: Path) -> Optional[str]:
    """Submit an sbatch script, return the job ID string or None on failure."""
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["sbatch"], input=script, text=True, capture_output=True
    )
    if result.returncode == 0:
        # Output is "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted: {result.stdout.strip()}")
        return job_id
    else:
        print(f"  Failed: {result.stderr.strip()}")
        return None


def submit_sequential_eval(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    calibration_path: Path,
    qos: str = "default",
    dry_run: bool = False,
) -> None:
    """Submit a sequential AFHP evaluation job using pre-computed calibration."""
    seq_script = build_sequential_eval_sbatch_command(
        job_name, eval_args, conda_env, log_dir, calibration_path, qos=qos
    )

    if dry_run:
        print(f"=== Sequential eval job: {job_name}_seq ===")
        print(seq_script)
        print()
        return

    if not _ensure_calibration_exists(job_name, calibration_path):
        return

    print(f"  Submitting sequential eval job using calibration {calibration_path}...")
    _sbatch_submit(seq_script, log_dir)


def submit_parallel_bins(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    calibration_path: Path,
    num_bins: int,
    qos: str = "default",
    dry_run: bool = False,
) -> None:
    """Submit a bin array AFHP evaluation job using pre-computed calibration."""
    bin_script = build_bin_array_sbatch_command(
        job_name,
        eval_args,
        conda_env,
        log_dir,
        calibration_path,
        num_bins,
        qos=qos,
    )

    if dry_run:
        print(f"=== Bin array job: {job_name}_bin (0-{num_bins - 1}) ===")
        print(bin_script)
        print()
        return

    if not _ensure_calibration_exists(job_name, calibration_path):
        return

    print(f"  Submitting bin array job (0-{num_bins - 1}) using calibration {calibration_path}...")
    _sbatch_submit(bin_script, log_dir)


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
        "--env", required=True, choices=ENVS, help="Environment to evaluate"
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
        "--num-bins",
        type=int,
        default=EVAL_DEFAULTS["num_bins"],
        help=f"Number of AFHP bins to evaluate (default: {EVAL_DEFAULTS['num_bins']})",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Submit a single sequential AFHP eval job that reuses a pre-computed "
            "calibration file. By default, submits a SLURM array of --num-bins bin "
            "jobs using the same calibration file."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override wandb project name",
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

    # Get config file path
    config_file = METHOD_CONFIGS[args.method]
    config_path = f"configs/eval/{args.env}/{config_file}"

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
            svdd_model_path = get_svdd_model_path(
                args.env, exp_id, args.method, svdd_base_path
            )
            cp_feature = "obs" if args.method == "svdd-image" else "hidden"

        # Get ensemble-specific settings if needed
        ensemble_members = None
        if args.method in ENSEMBLE_METHODS:
            ensemble_members = get_ensemble_member_paths(
                args.env, exp_id, checkpoint_base_path, args.num_ensemble_members
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
            svdd_policy_name = get_svdd_policy_name(args.env, exp_id, args.method)
            print(
                f"Warning: exp{exp_id} SVDD model not found at {svdd_base_path}/{svdd_policy_name}/trained.joblib"
            )
            missing = True

        if args.method in ENSEMBLE_METHODS and ensemble_members:
            for i, member_path in enumerate(ensemble_members):
                if member_path is None:
                    env_folder = get_env_folder(args.env)
                    print(
                        f"Warning: exp{exp_id} ensemble member m{i} not found at "
                        f"{checkpoint_base_path}/{env_folder}/ensembles/icml2_ensemble_{args.env}_exp{exp_id}_m{i}/"
                    )
                    missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing files\n")
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
            "num_bins": args.num_bins,
            "wandb_project": args.wandb_project,
            "level_seeds_file": str(level_seeds_file),
            "svdd_model_path": svdd_model_path,
            "cp_feature": cp_feature,
            "ensemble_members": ensemble_members,
            **checkpoints,
        }
        calibration_path = resolve_calibration_path(job_name, eval_args)
        print(f"Calibration path: {calibration_path}")

        if args.sequential:
            submit_sequential_eval(
                job_name,
                eval_args,
                args.conda_env,
                log_dir,
                calibration_path,
                args.qos,
                dry_run=args.dry_run,
            )
        else:
            submit_parallel_bins(
                job_name=job_name,
                eval_args=eval_args,
                conda_env=args.conda_env,
                log_dir=log_dir,
                calibration_path=calibration_path,
                num_bins=args.num_bins,
                qos=args.qos,
                dry_run=args.dry_run,
            )

    return 0


if __name__ == "__main__":
    exit(main())

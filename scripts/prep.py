"""Shared helpers for building and submitting SLURM prep jobs.

These helpers are used by `scripts/run_eval.py` (to auto-insert a calibrate
job before the eval array on cache miss) and by `scripts/train_svdd.py` (to
submit the SVDD gather -> train chain).
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_GATHER_CONFIG = "configs/procgen_gather.yaml"
DEFAULT_TRAIN_CONFIG = "configs/procgen_ood.yaml"
DEFAULT_NUM_ROLLOUTS = 64

SLURM_PREP_CONFIG: Dict[str, str] = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "mem": "256G",
    "cpus-per-task": "16",
}

EXP_ID_TO_SEED: Dict[int, int] = {
    0: 6033,
    1: 1,
    2: 2,
}


def sbatch_submit(script: str, log_dir: Path) -> Optional[str]:
    """Submit an sbatch script, returning the job ID on success."""
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if result.returncode == 0:
        return result.stdout.strip().split()[-1]

    print(f"Submission failed:\n{result.stderr}")
    return None


def build_sbatch_script(
    job_name: str,
    python_cmd: str,
    conda_env: str,
    output_root: Path,
    log_dir: Path,
    *,
    dependency_job_id: Optional[str] = None,
    qos: str = "default",
) -> str:
    """Build an sbatch script for one prep phase."""
    slurm_config = SLURM_PREP_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())
    dependency_line = (
        f"#SBATCH --dependency=afterok:{dependency_job_id}"
        if dependency_job_id is not None
        else ""
    )

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{dependency_line}
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

eval "$(conda shell.bash hook)"
conda activate {conda_env}
export SM_OUTPUT_DIR="{output_root}"
srun {slurm_args} {python_cmd}
"""


def build_base_python_args(
    script: str,
    config_path: str,
    run_name: str,
    experiment_group: str,
    checkpoints: dict,
    level_seeds_file: Path,
    *,
    feature_type: Optional[str] = None,
    file_name: Optional[str] = None,
    ensemble_members: Optional[List[Optional[str]]] = None,
) -> List[str]:
    """Build the shared python CLI args used by calibrate/gather/train commands."""
    args = [
        f"python {script}",
        f"-c {config_path}",
        f"-n {run_name}",
        f"-experiment_group {experiment_group}",
        f"-sim {checkpoints['sim']}",
        f"-weak {checkpoints['weak']}",
        f"-strong {checkpoints['strong']}",
        f"-level_seeds_file {level_seeds_file}",
    ]
    if feature_type:
        args.append(f"-cp_feature {feature_type}")
    if file_name:
        args.append(f"-f_n {file_name}")
    if ensemble_members:
        args.append("-cp_ensemble_members")
        for member_path in ensemble_members:
            if member_path is not None:
                args.append(str(member_path))
    return args


def build_calibration_command(
    coordination_artifact_dir: Path,
    experiment_group: str,
    config_path: str,
    checkpoints: dict,
    level_seeds_file: Path,
    *,
    feature_type: Optional[str] = None,
    file_name: Optional[str] = None,
    ensemble_members: Optional[List[Optional[str]]] = None,
) -> str:
    """Build a ``python calibrate_afhp.py ...`` command."""
    args = build_base_python_args(
        "calibrate_afhp.py",
        config_path,
        coordination_artifact_dir.name,
        experiment_group,
        checkpoints,
        level_seeds_file,
        feature_type=feature_type,
        file_name=file_name,
        ensemble_members=ensemble_members,
    )
    args.append(f"--coordination_artifact_dir {coordination_artifact_dir}")
    return " ".join(args)


def build_gather_command(
    coordination_artifact_dir: Path,
    experiment_group: str,
    env: str,
    checkpoints: dict,
    level_seeds_file: Path,
    seed: int,
    gather_config: str,
    num_rollouts: int,
    query_cost: float,
) -> str:
    """Build a ``python gather_rollouts.py ...`` command for SVDD prep."""
    args = [
        "python gather_rollouts.py",
        "-wandb_mode offline",
        f"-c {gather_config}",
        f"-n {coordination_artifact_dir.name}",
        f"--experiment_group {experiment_group}",
        f"-en {env}",
        "-random_percent 0",
        f"-sim {checkpoints['sim']}",
        f"-weak {checkpoints['weak']}",
        f"-strong {checkpoints['strong']}",
        f"-num_rollouts={num_rollouts}",
        "-use_bg=True",
        f"-seed {seed}",
        f"-level_seeds_file {level_seeds_file}",
        f"-query_cost {query_cost}",
    ]
    return " ".join(args)


def build_train_command(
    coordination_artifact_dir: Path,
    experiment_group: str,
    env: str,
    checkpoints: dict,
    feature_type: str,
    seed: int,
    train_config: str,
    num_rollouts: int,
    query_cost: float,
) -> str:
    """Build a ``python train.py ...`` command for SVDD prep."""
    args = [
        "python train.py",
        f"-wandb_group {experiment_group}",
        f"-c {train_config}",
        f"-n {coordination_artifact_dir.name}",
        f"-en {env}",
        f"-sim {checkpoints['sim']}",
        f"-weak {checkpoints['weak']}",
        f"-strong {checkpoints['strong']}",
        "-cp_method DeepSVDD",
        f"-cp_feature {feature_type}",
        f"-rollout_dir {coordination_artifact_dir.name}",
        f"-num_rollouts {num_rollouts}",
        "-wandb",
        f"-query_cost {query_cost}",
        f"-seed {seed}",
        "-over",
    ]
    return " ".join(args)

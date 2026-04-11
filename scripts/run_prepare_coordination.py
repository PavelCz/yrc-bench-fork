#!/usr/bin/env python3
"""
Prepare coordination-policy artifacts via SLURM.

For the methods currently supported by `scripts/run_eval.py`:
- `svdd-image`, `svdd-latent`: gather rollouts -> train DeepSVDD -> calibrate
- all other supported methods: calibrate only
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from YRC.core.artifacts import (  # noqa: E402
    default_coordination_artifact_root,
    resolve_calibration_path,
    resolve_coordination_artifact_dir,
)
from scripts.common import (  # noqa: E402
    DEFAULT_NUM_ENSEMBLE_MEMBERS,
    ENSEMBLE_METHODS,
    ENVS,
    METHOD_CONFIGS,
    METHOD_NAMES,
    SERVER_PATHS,
    SVDD_METHODS,
    get_checkpoints,
    get_ensemble_member_paths,
    get_svdd_feature_type,
)

DEFAULT_CONDA_ENV = "ood-stable"
DEFAULT_GATHER_CONFIG = "configs/procgen_gather.yaml"
DEFAULT_TRAIN_CONFIG = "configs/procgen_ood.yaml"
DEFAULT_NUM_ROLLOUTS = 64
SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "mem": "256G",
    "cpus-per-task": "16",
}

EXP_ID_TO_SEED = {
    0: 6033,
    1: 1,
    2: 2,
}


@dataclass(frozen=True)
class PreparePlan:
    requires_rollouts: bool
    requires_training: bool
    requires_calibration: bool = True


def get_prepare_plan(method: str) -> PreparePlan:
    """Return which preparation phases are required for a method."""
    if method in SVDD_METHODS:
        return PreparePlan(requires_rollouts=True, requires_training=True)
    if method in METHOD_CONFIGS:
        return PreparePlan(requires_rollouts=False, requires_training=False)
    raise ValueError(f"Unsupported method: {method}")


def _sbatch_submit(script: str, log_dir: Path) -> Optional[str]:
    """Submit an sbatch script, returning the job ID on success."""
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if result.returncode == 0:
        return result.stdout.strip().split()[-1]

    print(f"Submission failed:\n{result.stderr}")
    return None


def _build_base_python_args(
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
    args = _build_base_python_args(
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
    """Build an sbatch script for one preparation phase."""
    slurm_config = SLURM_CONFIG.copy()
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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare coordination-policy artifacts via SLURM"
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
        default="snoopy",
        help="Server to use for paths (default: snoopy)",
    )
    parser.add_argument(
        "--qos",
        choices=["default", "high"],
        default="default",
        help="SLURM QOS to use (default: default)",
    )
    parser.add_argument(
        "--env", required=True, choices=ENVS, help="Environment to prepare for"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHOD_CONFIGS.keys()),
        help="Coordination method to prepare",
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
        "--num-rollouts",
        type=int,
        default=DEFAULT_NUM_ROLLOUTS,
        help=f"Number of rollouts for trainable methods (default: {DEFAULT_NUM_ROLLOUTS})",
    )
    parser.add_argument(
        "--query-cost",
        type=float,
        default=0.0,
        help="Query cost to pass through to gather/train/calibration",
    )
    parser.add_argument(
        "--gather-config",
        default=DEFAULT_GATHER_CONFIG,
        help="Gather-rollouts config for trainable methods",
    )
    parser.add_argument(
        "--train-config",
        default=DEFAULT_TRAIN_CONFIG,
        help="Training config for trainable methods",
    )
    parser.add_argument(
        "--coordination-artifact-root",
        default=None,
        help=(
            "Base directory for coordination-method artifacts. "
            "Defaults to a sibling of the acting-policy root."
        ),
    )
    parser.add_argument(
        "--num-ensemble-members",
        type=int,
        default=DEFAULT_NUM_ENSEMBLE_MEMBERS,
        help=(
            "Number of additional ensemble members for ensemble methods "
            f"(default: {DEFAULT_NUM_ENSEMBLE_MEMBERS})"
        ),
    )
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    args = parser.parse_args()

    if not Path(args.gather_config).exists():
        print(f"Error: Gather config not found: {args.gather_config}")
        return 1
    if not Path(args.train_config).exists():
        print(f"Error: Train config not found: {args.train_config}")
        return 1

    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    seeds_base_path = paths["seeds_base"]
    coordination_root = (
        Path(args.coordination_artifact_root)
        if args.coordination_artifact_root is not None
        else default_coordination_artifact_root(Path(checkpoint_base_path))
    )

    eval_config = f"configs/eval/{args.env}/{METHOD_CONFIGS[args.method]}"
    if not Path(eval_config).exists():
        print(f"Error: Eval config not found: {eval_config}")
        return 1

    plan = get_prepare_plan(args.method)
    method_name = METHOD_NAMES[args.method]
    feature_type = (
        get_svdd_feature_type(args.method) if args.method in SVDD_METHODS else None
    )

    print(f"Conda env: {args.conda_env}")
    print(f"Coordination artifact root: {coordination_root}")

    for exp_id in args.exp_ids:
        checkpoints = get_checkpoints(args.env, exp_id, checkpoint_base_path)
        if args.sim:
            checkpoints["sim"] = args.sim
        if args.weak:
            checkpoints["weak"] = args.weak
        if args.strong:
            checkpoints["strong"] = args.strong

        level_seeds_file = Path(seeds_base_path) / f"{exp_id}.json"
        experiment_group = f"{args.prefix}_{args.env}_exp{exp_id}"
        coordination_artifact_dir = resolve_coordination_artifact_dir(
            args.env,
            exp_id,
            method_name,
            experiment_group,
            coordination_root=coordination_root,
        )
        calibration_path = resolve_calibration_path(coordination_artifact_dir)
        log_dir = coordination_artifact_dir / "slurm"
        seed = EXP_ID_TO_SEED.get(exp_id, exp_id)

        ensemble_members: Optional[List[Optional[str]]] = None
        if args.method in ENSEMBLE_METHODS:
            ensemble_members = get_ensemble_member_paths(
                args.env,
                exp_id,
                checkpoint_base_path,
                args.num_ensemble_members,
            )

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

        if ensemble_members:
            for i, member_path in enumerate(ensemble_members):
                if member_path is None:
                    print(f"Warning: exp{exp_id} ensemble member m{i} not found")
                    missing = True

        if missing:
            print(f"Skipping exp{exp_id} due to missing files\n")
            continue

        print(f"=== exp{exp_id} ===")
        print(f"  Method: {args.method}")
        print(f"  Coordination artifact dir: {coordination_artifact_dir}")
        print(f"  Calibration path: {calibration_path}")

        gather_job_id = None
        train_job_id = None

        if plan.requires_rollouts:
            gather_cmd = build_gather_command(
                coordination_artifact_dir,
                experiment_group,
                args.env,
                checkpoints,
                level_seeds_file,
                seed,
                args.gather_config,
                args.num_rollouts,
                args.query_cost,
            )
            gather_script = build_sbatch_script(
                f"{method_name}_{args.env}_exp{exp_id}_gather",
                gather_cmd,
                args.conda_env,
                coordination_artifact_dir.parent,
                log_dir,
                qos=args.qos,
            )
            if args.dry_run:
                print("--- Gather job ---")
                print(gather_script)
            else:
                print("  Submitting gather job...")
                gather_job_id = _sbatch_submit(gather_script, log_dir)
                if gather_job_id is None:
                    print(f"  Aborting preparation for exp{exp_id}.")
                    continue

        if plan.requires_training:
            train_cmd = build_train_command(
                coordination_artifact_dir,
                experiment_group,
                args.env,
                checkpoints,
                feature_type=feature_type or "obs",
                seed=seed,
                train_config=args.train_config,
                num_rollouts=args.num_rollouts,
                query_cost=args.query_cost,
            )
            train_script = build_sbatch_script(
                f"{method_name}_{args.env}_exp{exp_id}_train",
                train_cmd,
                args.conda_env,
                coordination_artifact_dir.parent,
                log_dir,
                dependency_job_id=gather_job_id,
                qos=args.qos,
            )
            if args.dry_run:
                print("--- Train job ---")
                print(train_script)
            else:
                print("  Submitting train job...")
                train_job_id = _sbatch_submit(train_script, log_dir)
                if train_job_id is None:
                    print(f"  Aborting preparation for exp{exp_id}.")
                    continue

        if plan.requires_calibration:
            calibration_cmd = build_calibration_command(
                coordination_artifact_dir,
                experiment_group,
                eval_config,
                checkpoints,
                level_seeds_file,
                feature_type=feature_type,
                file_name="trained.joblib" if args.method in SVDD_METHODS else None,
                ensemble_members=ensemble_members,
            )
            calibration_script = build_sbatch_script(
                f"{method_name}_{args.env}_exp{exp_id}_calib",
                calibration_cmd,
                args.conda_env,
                coordination_artifact_dir.parent,
                log_dir,
                dependency_job_id=train_job_id,
                qos=args.qos,
            )
            if args.dry_run:
                print("--- Calibration job ---")
                print(calibration_script)
                print()
            else:
                print("  Submitting calibration job...")
                calib_job_id = _sbatch_submit(calibration_script, log_dir)
                if calib_job_id is None:
                    print(f"  Calibration submission failed for exp{exp_id}.")
                    continue
                print(f"  Submitted calibration job {calib_job_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Submit AFHP evaluation jobs via SLURM.

High-level flow:
1. Parse CLI arguments and resolve shared paths/config.
2. For each requested experiment id, collect and validate inputs.
3. Ensure calibration exists (or submit a calibration job first).
4. Submit the AFHP bin array job for that experiment.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import tyro

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (  # noqa: E402
    DEFAULT_NUM_ENSEMBLE_MEMBERS,
    ENSEMBLE_METHODS,
    METHOD_CONFIGS,
    METHOD_NAMES,
    SERVER_PATHS,
    SVDD_METHODS,
    get_env_folder,
    get_checkpoints,
    get_ensemble_member_paths,
    get_svdd_policy_name,
    get_svdd_model_path,
)
from scripts.prep import (  # noqa: E402
    build_calibration_command,
    build_sbatch_script,
)
from YRC.core.artifacts import (  # noqa: E402
    default_coordination_artifact_root,
    resolve_calibration_path,
    resolve_coordination_artifact_dir,
)

# Sentinel signalling the caller should skip this experiment id entirely
# (e.g. SVDD trained model missing).
_SKIP_EXP = object()
_DRY_RUN_CALIB_JOB_ID = "DRY_RUN_CALIB_JOB_ID"


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

EnvName = Literal["maze", "coinrun"]
QosName = Literal["default", "high"]
VideoFilterMode = Literal["any", "all"]


@dataclass(frozen=True)
class SharedEvalContext:
    """Invocation-wide settings shared across all experiment ids."""

    checkpoint_base_path: str
    seeds_base_path: str
    svdd_base_path: str
    config_path: str
    coordination_root: Path
    log_dir: Path


@dataclass(frozen=True)
class ExperimentPlan:
    """All resolved inputs needed to submit evaluation for one experiment id."""

    exp_id: int
    job_name: str
    experiment_group: str
    checkpoints: Dict[str, str]
    level_seeds_file: Path
    svdd_model_path: Optional[str]
    cp_feature: Optional[str]
    ensemble_members: Optional[List[Optional[str]]]
    coordination_artifact_dir: Path
    calibration_path: Path
    eval_args: Dict[str, object]


@dataclass(frozen=True)
class RunEvalCliArgs:
    """CLI arguments for `scripts/run_eval.py`."""

    env: EnvName
    method: str
    prefix: str
    dry_run: bool = False
    conda_env: str = DEFAULT_CONDA_ENV
    server: str = "chai"
    qos: QosName = "default"
    exp_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_levels: int = EVAL_DEFAULTS["num_levels"]
    video_episodes: int = EVAL_DEFAULTS["video_episodes_to_collect"]
    video_filter: str = EVAL_DEFAULTS["video_filter"]
    cp_rolling_average: str = EVAL_DEFAULTS["cp_rolling_average"]
    video_logging_mode: str = EVAL_DEFAULTS["video_logging_mode"]
    video_filter_mode: VideoFilterMode = EVAL_DEFAULTS["video_filter_mode"]
    num_bins: int = EVAL_DEFAULTS["num_bins"]
    wandb_project: Optional[str] = None
    coordination_artifact_root: Optional[Path] = None
    sim: Optional[str] = None
    weak: Optional[str] = None
    strong: Optional[str] = None
    num_ensemble_members: int = DEFAULT_NUM_ENSEMBLE_MEMBERS


def main():
    args = parse_args()
    context = resolve_shared_eval_context(args)
    if context is None:
        return 1

    print_submission_overview(args, context)
    run_requested_evaluations(args, context)
    return 0


def parse_args() -> RunEvalCliArgs:
    """Parse and validate the Tyro CLI for this script."""
    args = tyro.cli(RunEvalCliArgs)
    validate_cli_args(args)
    return args


def validate_cli_args(args: RunEvalCliArgs) -> None:
    """Validate CLI values against shared repo constants."""
    _validate_choice("server", args.server, SERVER_PATHS.keys())
    _validate_choice("method", args.method, METHOD_CONFIGS.keys())
    if args.num_ensemble_members < 1:
        raise SystemExit("Error: --num-ensemble-members must be >= 1.")


def _validate_choice(name: str, value: str, allowed: Iterable[str]) -> None:
    allowed_values = sorted(allowed)
    if value not in allowed_values:
        raise SystemExit(f"Error: invalid {name} {value!r}. Allowed: {allowed_values}")


def run_requested_evaluations(args: RunEvalCliArgs, context: SharedEvalContext) -> None:
    """Plan and submit all requested experiment ids."""
    for exp_id in args.exp_ids:
        plan = plan_experiment_submission(args, context, exp_id)
        if plan is None:
            continue
        submit_experiment_plan(args, context, plan)


def plan_experiment_submission(
    args: RunEvalCliArgs, context: SharedEvalContext, exp_id: int
) -> Optional[ExperimentPlan]:
    """Resolve and validate everything needed to submit one experiment id."""
    checkpoints = resolve_checkpoints_for_experiment(
        args, exp_id, context.checkpoint_base_path
    )
    level_seeds_file = Path(context.seeds_base_path) / f"{exp_id}.json"
    svdd_model_path, cp_feature = resolve_svdd_settings(
        args, exp_id, context.svdd_base_path
    )
    ensemble_members = resolve_ensemble_members(
        args, exp_id, context.checkpoint_base_path
    )

    if not validate_experiment_inputs(
        args,
        exp_id,
        checkpoints=checkpoints,
        level_seeds_file=level_seeds_file,
        svdd_model_path=svdd_model_path,
        ensemble_members=ensemble_members,
        checkpoint_base_path=context.checkpoint_base_path,
        svdd_base_path=context.svdd_base_path,
    ):
        print(f"Skipping exp{exp_id} due to missing files\n")
        return None

    method_name = METHOD_NAMES[args.method]
    job_name = f"{args.env}_{method_name}_exp{exp_id}"
    experiment_group = f"{args.prefix}_{args.env}_exp{exp_id}"
    coordination_artifact_dir = resolve_coordination_artifact_dir(
        args.env,
        exp_id,
        method_name,
        experiment_group,
        coordination_root=context.coordination_root,
    )
    calibration_path = resolve_calibration_path(coordination_artifact_dir)
    eval_args = build_eval_args(
        args,
        context,
        job_name=job_name,
        experiment_group=experiment_group,
        checkpoints=checkpoints,
        level_seeds_file=level_seeds_file,
        svdd_model_path=svdd_model_path,
        cp_feature=cp_feature,
        ensemble_members=ensemble_members,
    )
    return ExperimentPlan(
        exp_id=exp_id,
        job_name=job_name,
        experiment_group=experiment_group,
        checkpoints=checkpoints,
        level_seeds_file=level_seeds_file,
        svdd_model_path=svdd_model_path,
        cp_feature=cp_feature,
        ensemble_members=ensemble_members,
        coordination_artifact_dir=coordination_artifact_dir,
        calibration_path=calibration_path,
        eval_args=eval_args,
    )


def submit_experiment_plan(
    args: RunEvalCliArgs, context: SharedEvalContext, plan: ExperimentPlan
) -> None:
    """Submit calibration if needed, then submit the AFHP bin array job."""
    print(f"Calibration path: {plan.calibration_path}")
    dependency_job_id = _maybe_submit_calibration_job(
        method=args.method,
        calibration_path=plan.calibration_path,
        coordination_artifact_dir=plan.coordination_artifact_dir,
        experiment_group=plan.experiment_group,
        config_path=context.config_path,
        checkpoints=plan.checkpoints,
        level_seeds_file=plan.level_seeds_file,
        cp_feature=plan.cp_feature,
        svdd_model_path=plan.svdd_model_path,
        ensemble_members=plan.ensemble_members,
        conda_env=args.conda_env,
        log_dir=context.log_dir,
        qos=args.qos,
        env=args.env,
        exp_id=plan.exp_id,
        dry_run=args.dry_run,
    )
    if dependency_job_id is _SKIP_EXP:
        print(f"Skipping exp{plan.exp_id}.\n")
        return

    submit_parallel_bins(
        job_name=plan.job_name,
        eval_args=plan.eval_args,
        conda_env=args.conda_env,
        log_dir=context.log_dir,
        calibration_path=plan.calibration_path,
        num_bins=args.num_bins,
        qos=args.qos,
        dry_run=args.dry_run,
        dependency_job_id=dependency_job_id,
    )


def resolve_shared_eval_context(
    args: RunEvalCliArgs,
) -> Optional[SharedEvalContext]:
    """Resolve invocation-wide paths and validate the eval config exists."""
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    seeds_base_path = paths["seeds_base"]
    svdd_base_path = paths["svdd_base"]
    coordination_root = (
        Path(args.coordination_artifact_root)
        if args.coordination_artifact_root is not None
        else default_coordination_artifact_root(Path(checkpoint_base_path))
    )
    config_file = METHOD_CONFIGS[args.method]
    config_path = f"configs/eval/{args.env}/{config_file}"
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return None

    wandb_project = args.wandb_project or "default"
    log_dir = (
        Path(paths["log_base"]) / wandb_project / args.prefix / date.today().isoformat()
    )
    return SharedEvalContext(
        checkpoint_base_path=checkpoint_base_path,
        seeds_base_path=seeds_base_path,
        svdd_base_path=svdd_base_path,
        config_path=config_path,
        coordination_root=coordination_root,
        log_dir=log_dir,
    )


def print_submission_overview(args: RunEvalCliArgs, context: SharedEvalContext) -> None:
    """Print the shared settings for this invocation."""
    print(f"Conda env: {args.conda_env}")
    print(f"Coordination artifact root: {context.coordination_root}")
    print(f"Log dir: {context.log_dir}")

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {context.config_path}")
        print(f"Environment: {args.env}")
        print(f"Method: {args.method}")
        print(f"Prefix: {args.prefix}")
        print()


def resolve_checkpoints_for_experiment(
    args: RunEvalCliArgs, exp_id: int, checkpoint_base_path: str
) -> Dict[str, str]:
    """Resolve acting-policy checkpoints, applying any CLI overrides."""
    checkpoints = get_checkpoints(args.env, exp_id, checkpoint_base_path)
    if args.sim:
        checkpoints["sim"] = args.sim
    if args.weak:
        checkpoints["weak"] = args.weak
    if args.strong:
        checkpoints["strong"] = args.strong
    return checkpoints


def resolve_svdd_settings(
    args: RunEvalCliArgs, exp_id: int, svdd_base_path: str
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve SVDD-specific model path and feature type."""
    if args.method not in SVDD_METHODS:
        return None, None

    svdd_model_path = get_svdd_model_path(args.env, exp_id, args.method, svdd_base_path)
    cp_feature = "obs" if args.method == "svdd-image" else "hidden"
    return svdd_model_path, cp_feature


def resolve_ensemble_members(
    args: RunEvalCliArgs, exp_id: int, checkpoint_base_path: str
) -> Optional[List[Optional[str]]]:
    """Resolve ensemble member checkpoints for ensemble methods."""
    if args.method not in ENSEMBLE_METHODS:
        return None
    return get_ensemble_member_paths(
        args.env, exp_id, checkpoint_base_path, args.num_ensemble_members
    )


def validate_experiment_inputs(
    args: RunEvalCliArgs,
    exp_id: int,
    *,
    checkpoints: Dict[str, str],
    level_seeds_file: Path,
    svdd_model_path: Optional[str],
    ensemble_members: Optional[List[Optional[str]]],
    checkpoint_base_path: str,
    svdd_base_path: str,
) -> bool:
    """Return True when all required inputs for one experiment are present."""
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
        print(
            f"Error: exp{exp_id} SVDD model not found at "
            f"{svdd_base_path}/{svdd_policy_name}/trained.joblib. "
            f"Run `python scripts/train_svdd.py --env {args.env} "
            f"--method {args.method} --prefix {args.prefix} --exp-ids {exp_id}` first."
        )
        missing = True

    if args.method in ENSEMBLE_METHODS and ensemble_members:
        env_folder = get_env_folder(args.env)
        for i, member_path in enumerate(ensemble_members):
            if member_path is None:
                print(
                    f"Warning: exp{exp_id} ensemble member m{i} not found at "
                    f"{checkpoint_base_path}/{env_folder}/ensembles/"
                    f"icml2_ensemble_{args.env}_exp{exp_id}_m{i}/"
                )
                missing = True

    return not missing


def build_eval_args(
    args: RunEvalCliArgs,
    context: SharedEvalContext,
    *,
    job_name: str,
    experiment_group: str,
    checkpoints: Dict[str, str],
    level_seeds_file: Path,
    svdd_model_path: Optional[str],
    cp_feature: Optional[str],
    ensemble_members: Optional[List[Optional[str]]],
) -> Dict[str, object]:
    """Build the argument bundle passed to downstream job builders."""
    return {
        "config": context.config_path,
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


def _maybe_submit_calibration_job(
    *,
    method: str,
    calibration_path: Path,
    coordination_artifact_dir: Path,
    experiment_group: str,
    config_path: str,
    checkpoints: dict,
    level_seeds_file: Path,
    cp_feature: Optional[str],
    svdd_model_path: Optional[str],
    ensemble_members: Optional[List[Optional[str]]],
    conda_env: str,
    log_dir: Path,
    qos: str,
    env: str,
    exp_id: int,
    dry_run: bool,
):
    """Insert a calibration job before the eval array on cache miss.

    Returns one of:
    - ``None`` on cache hit (calibration already exists; eval runs without a dep).
    - A job ID string on successful submission (eval should depend on it).
    - The ``_SKIP_EXP`` sentinel if this experiment id cannot be run (e.g. SVDD
      trained model is missing) and the caller should skip it entirely.
    """
    if calibration_path.exists():
        print(f"  Calibration cache hit: {calibration_path}")
        return None

    print(f"  Calibration missing at {calibration_path}")

    if method in SVDD_METHODS:
        if svdd_model_path is None or not Path(svdd_model_path).exists():
            print(
                f"  Error: SVDD trained model missing for exp{exp_id}. "
                f"Run `python scripts/train_svdd.py --env {env} "
                f"--method {method} --prefix <prefix> --exp-ids {exp_id}` first."
            )
            return _SKIP_EXP

    file_name = "trained.joblib" if method in SVDD_METHODS else None

    calibration_cmd = build_calibration_command(
        coordination_artifact_dir,
        experiment_group,
        config_path,
        checkpoints,
        level_seeds_file,
        feature_type=cp_feature,
        file_name=file_name,
        ensemble_members=ensemble_members,
    )
    job_name = f"{coordination_artifact_dir.name}_calib"
    calibration_script = build_sbatch_script(
        job_name,
        calibration_cmd,
        conda_env,
        coordination_artifact_dir.parent,
        log_dir,
        qos=qos,
    )

    if dry_run:
        print(f"=== Calibration job: {job_name} ===")
        print(calibration_script)
        return _DRY_RUN_CALIB_JOB_ID

    print("  Submitting calibration job...")
    job_id = _sbatch_submit(calibration_script, log_dir)
    if job_id is None:
        print(f"  Calibration submission failed for exp{exp_id}.")
        return _SKIP_EXP
    print(f"  Submitted calibration job {job_id}")
    return job_id


def submit_parallel_bins(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    calibration_path: Path,
    num_bins: int,
    qos: str = "default",
    dry_run: bool = False,
    dependency_job_id: Optional[str] = None,
) -> None:
    """Submit a bin array AFHP evaluation job.

    If ``dependency_job_id`` is provided, the eval array waits for that job
    (typically a calibrate job inserted on cache miss) via afterok.
    """
    bin_script = build_bin_array_sbatch_command(
        job_name,
        eval_args,
        conda_env,
        log_dir,
        calibration_path,
        num_bins,
        qos=qos,
        dependency_job_id=dependency_job_id,
    )

    if dry_run:
        print(f"=== Bin array job: {job_name}_bin (0-{num_bins - 1}) ===")
        print(bin_script)
        print()
        return

    dep_note = (
        f" (depends on calib job {dependency_job_id})"
        if dependency_job_id is not None
        else ""
    )
    print(
        f"  Submitting bin array job (0-{num_bins - 1}) using calibration "
        f"{calibration_path}{dep_note}..."
    )
    _sbatch_submit(bin_script, log_dir)


def build_bin_array_sbatch_command(
    job_name: str,
    eval_args: dict,
    conda_env: str,
    log_dir: Path,
    calibration_path: Path,
    num_bins: int,
    qos: str = "default",
    dependency_job_id: Optional[str] = None,
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
{_dependency_line(dependency_job_id)}
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

echo "Bin $SLURM_ARRAY_TASK_ID / {num_bins} for {job_name}"
eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""


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


def _dependency_line(dependency_job_id: Optional[str]) -> str:
    if dependency_job_id is None:
        return ""
    return f"#SBATCH --dependency=afterok:{dependency_job_id}"


def _sbatch_submit(script: str, log_dir: Path) -> Optional[str]:
    """Submit an sbatch script, return the job ID string or None on failure."""
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if result.returncode == 0:
        # Output is "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted: {result.stdout.strip()}")
        return job_id
    print(f"  Failed: {result.stderr.strip()}")
    return None


if __name__ == "__main__":
    exit(main())

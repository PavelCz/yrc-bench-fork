#!/usr/bin/env python3
"""
Train SVDD coordination models via SLURM.

Submits a ``gather_rollouts -> train`` chain with ``--dependency=afterok``
for each requested experiment id. This is the explicit training entrypoint
for SVDD methods and never runs calibration or eval as a side effect.

For non-SVDD methods, calibration is auto-inserted by ``scripts/run_eval.py``
on cache miss. This script rejects non-SVDD methods.
"""

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from YRC.core.artifacts import (  # noqa: E402
    default_coordination_artifact_root,
    resolve_coordination_artifact_dir,
)
from scripts.common import (  # noqa: E402
    ENVS,
    METHOD_NAMES,
    SERVER_PATHS,
    SVDD_METHODS,
    get_checkpoints,
    get_svdd_feature_type,
)
from scripts.prep import (  # noqa: E402
    DEFAULT_GATHER_CONFIG,
    DEFAULT_NUM_ROLLOUTS,
    DEFAULT_TRAIN_CONFIG,
    EXP_ID_TO_SEED,
    build_gather_command,
    build_sbatch_script,
    build_train_command,
    sbatch_submit,
)

DEFAULT_CONDA_ENV = "ood-stable"


def _print_script(phase: str, script: str) -> None:
    print(f"--- {phase} ---")
    print(script)


def submit_svdd_prep_for_exp(
    *,
    env: str,
    exp_id: int,
    method: str,
    prefix: str,
    checkpoint_base_path: str,
    seeds_base_path: str,
    coordination_root: Path,
    conda_env: str,
    qos: str,
    num_rollouts: int,
    query_cost: float,
    gather_config: str,
    train_config: str,
    checkpoint_overrides: dict,
    dry_run: bool,
) -> bool:
    """Submit the gather -> train chain for one experiment id.

    Returns ``True`` if submission succeeded (or dry-run printed), ``False``
    if the experiment id was skipped due to missing inputs or a failed
    submission.
    """
    checkpoints = get_checkpoints(env, exp_id, checkpoint_base_path)
    for name, override in checkpoint_overrides.items():
        if override is not None:
            checkpoints[name] = override

    level_seeds_file = Path(seeds_base_path) / f"{exp_id}.json"

    missing = False
    for name, path in checkpoints.items():
        if not Path(path).exists():
            print(f"Warning: exp{exp_id} {name} checkpoint not found: {path}")
            missing = True
    if not level_seeds_file.exists():
        print(f"Warning: exp{exp_id} level seeds file not found: {level_seeds_file}")
        missing = True
    if missing:
        print(f"Skipping exp{exp_id} due to missing files\n")
        return False

    method_name = METHOD_NAMES[method]
    feature_type = get_svdd_feature_type(method)
    experiment_group = f"{prefix}_{env}_exp{exp_id}"
    coordination_artifact_dir = resolve_coordination_artifact_dir(
        env,
        exp_id,
        method_name,
        experiment_group,
        coordination_root=coordination_root,
    )
    log_dir = coordination_artifact_dir / "slurm"
    seed = EXP_ID_TO_SEED.get(exp_id, exp_id)

    print(f"=== exp{exp_id} ===")
    print(f"  Method: {method}")
    print(f"  Coordination artifact dir: {coordination_artifact_dir}")

    gather_cmd = build_gather_command(
        coordination_artifact_dir,
        experiment_group,
        env,
        checkpoints,
        level_seeds_file,
        seed,
        gather_config,
        num_rollouts,
        query_cost,
    )
    gather_script = build_sbatch_script(
        f"{method_name}_{env}_exp{exp_id}_gather",
        gather_cmd,
        conda_env,
        coordination_artifact_dir.parent,
        log_dir,
        qos=qos,
    )

    gather_job_id: Optional[str] = None
    if dry_run:
        _print_script("Gather job", gather_script)
    else:
        print("  Submitting gather job...")
        gather_job_id = sbatch_submit(gather_script, log_dir)
        if gather_job_id is None:
            print(f"  Aborting training for exp{exp_id}.")
            return False

    train_cmd = build_train_command(
        coordination_artifact_dir,
        experiment_group,
        env,
        checkpoints,
        feature_type=feature_type,
        seed=seed,
        train_config=train_config,
        num_rollouts=num_rollouts,
        query_cost=query_cost,
    )
    train_script = build_sbatch_script(
        f"{method_name}_{env}_exp{exp_id}_train",
        train_cmd,
        conda_env,
        coordination_artifact_dir.parent,
        log_dir,
        dependency_job_id=gather_job_id,
        qos=qos,
    )

    if dry_run:
        _print_script("Train job", train_script)
        print()
        return True

    print("  Submitting train job...")
    train_job_id = sbatch_submit(train_script, log_dir)
    if train_job_id is None:
        print(f"  Train submission failed for exp{exp_id}.")
        return False
    print(f"  Submitted train job {train_job_id}")
    return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train SVDD coordination models via SLURM")
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
        "--env", required=True, choices=ENVS, help="Environment to train for"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(SVDD_METHODS),
        help="SVDD method to train",
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
        help=f"Number of rollouts for gather/train (default: {DEFAULT_NUM_ROLLOUTS})",
    )
    parser.add_argument(
        "--query-cost",
        type=float,
        default=0.0,
        help="Query cost to pass through to gather/train",
    )
    parser.add_argument(
        "--gather-config",
        default=DEFAULT_GATHER_CONFIG,
        help=f"Gather-rollouts config (default: {DEFAULT_GATHER_CONFIG})",
    )
    parser.add_argument(
        "--train-config",
        default=DEFAULT_TRAIN_CONFIG,
        help=f"Training config (default: {DEFAULT_TRAIN_CONFIG})",
    )
    parser.add_argument(
        "--coordination-artifact-root",
        default=None,
        help=(
            "Base directory for coordination-method artifacts. "
            "Defaults to a sibling of the acting-policy root."
        ),
    )
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    args = parser.parse_args()

    if args.method not in SVDD_METHODS:
        # argparse "choices" already enforces this; kept as a defensive guard.
        print(
            f"Error: train_svdd.py only supports SVDD methods, got {args.method!r}. "
            f"Allowed: {sorted(SVDD_METHODS)}"
        )
        return 1

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

    print(f"Conda env: {args.conda_env}")
    print(f"Coordination artifact root: {coordination_root}")

    checkpoint_overrides = {
        "sim": args.sim,
        "weak": args.weak,
        "strong": args.strong,
    }

    for exp_id in args.exp_ids:
        submit_svdd_prep_for_exp(
            env=args.env,
            exp_id=exp_id,
            method=args.method,
            prefix=args.prefix,
            checkpoint_base_path=checkpoint_base_path,
            seeds_base_path=seeds_base_path,
            coordination_root=coordination_root,
            conda_env=args.conda_env,
            qos=args.qos,
            num_rollouts=args.num_rollouts,
            query_cost=args.query_cost,
            gather_config=args.gather_config,
            train_config=args.train_config,
            checkpoint_overrides=checkpoint_overrides,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

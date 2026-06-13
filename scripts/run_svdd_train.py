#!/usr/bin/env python3
"""
Script to run DeepSVDD training jobs in parallel via SLURM sbatch.
"""

import netrc
import os
import shlex
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional

from common import (
    ENVS,
    EXP_ID_TO_SEED,
    SERVER_PATHS,
    get_checkpoints,
    get_eval_env_name,
)


def _get_wandb_api_key() -> Optional[str]:
    """Look up the wandb API key on the submitter side.

    Checks the WANDB_API_KEY environment variable first, then ~/.netrc for the
    api.wandb.ai entry. Returns None if neither is available.
    """
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key
    try:
        rc = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    auth = rc.authenticators("api.wandb.ai")
    if not auth:
        return None
    _login, _account, password = auth
    return password or None


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
    "rollouts_prefix": "dummy03",
    "rollout_max_levels": 1024,
    "svdd_val_levels": 64,
}

# Feature type choices
FEATURE_TYPES = ["obs", "hidden"]

# Optional DeepSVDD presets. See docs/image_svdd_collapse_bugs.md.
# Each entry maps a preset name to (prefix override, extra train_svdd.py
# CLI flags). The default path already includes the collapse bug fixes; the
# preset below additionally applies the paper-aligned regularisation settings.
VARIANT_PRESETS = {
    "paper-regularized": (
        "paper-regularized",
        [
            "-center_init_post_activation true",
            "-l2_regularizer 1e-6",
        ],
    ),
}


def get_rollout_dir(
    env: str, exp_id: int, rollouts_base_path: str, rollouts_prefix: str
) -> str:
    """Get the rollout directory path."""
    rollout_name = f"gather_{env}_exp{exp_id}"
    return str(Path(rollouts_base_path) / rollouts_prefix / env / rollout_name)


def get_svdd_env_name(env: str) -> str:
    """Map experiment env keys to the Procgen env used for SVDD training."""
    svdd_env_name = get_eval_env_name(env)
    if svdd_env_name == "maze":
        raise ValueError("Plain Procgen env 'maze' is not valid; use 'maze_afh'.")
    return svdd_env_name


def get_svdd_output_dir(svdd_base_path: str, prefix: str) -> Path:
    return Path(svdd_base_path) / prefix


def get_expected_model_file(svdd_output_dir: Path, job_name: str) -> Path:
    return svdd_output_dir / job_name / "trained.joblib"


def resolve_rollout_path_for_check(rollout_dir: str) -> Path:
    rollout_path = Path(rollout_dir)
    if rollout_path.is_absolute():
        return rollout_path
    return Path(os.getenv("SM_OUTPUT_DIR", "experiments")) / rollout_path


def build_sbatch_command(
    job_name: str,
    train_args: dict,
    log_dir: Path,
    *,
    redact_secrets: bool = False,
) -> str:
    """Build the sbatch command string.

    When ``redact_secrets`` is True, any embedded wandb API key is replaced
    with ``<redacted>``. Use this when printing the script (e.g. ``--dry-run``)
    so the key does not end up in stdout / shell history.
    """
    if train_args["env_name"] == "maze":
        raise ValueError("Plain Procgen env 'maze' is not valid; use 'maze_afh'.")

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
        f"-level_seeds_file {train_args['level_seeds_file']}",
        f"-svdd_val_levels {train_args['svdd_val_levels']}",
        f"-query_cost {train_args['query_cost']}",
        f"-seed {train_args['seed']}",
        "-over",
    ]
    if train_args.get("use_wandb", True):
        python_args.append("-wandb")
    else:
        # Force the second wandb.init in train_svdd.py to use disabled mode
        # so it doesn't try to authenticate.
        python_args.append("-wandb_mode disabled")
    if train_args["rollout_max_levels"] is not None:
        python_args.append(f"-rollout_max_levels {train_args['rollout_max_levels']}")
    for extra_flag in train_args.get("extra_python_args", []):
        python_args.append(extra_flag)
    python_cmd = " ".join(python_args)

    wandb_key = train_args.get("wandb_api_key") or None
    if wandb_key:
        embedded_key = "<redacted>" if redact_secrets else wandb_key
        # trap ensures WANDB_API_KEY is unset on any script exit (success,
        # failure, or interrupt). The key only lives for the duration of the
        # job process and never escapes into the post-job environment.
        wandb_env_block = (
            "trap 'unset WANDB_API_KEY' EXIT\n"
            f"export WANDB_API_KEY={shlex.quote(embedded_key)}\n"
        )
    else:
        wandb_env_block = ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items())}

eval "$(conda shell.bash hook)"
conda activate {CONDA_ENV}
{wandb_env_block}export SM_OUTPUT_DIR="{train_args["output_dir"]}"
srun {slurm_args} {python_cmd}
"""
    return sbatch_script


def submit_job(
    job_name: str, train_args: dict, log_dir: Path, dry_run: bool = False
) -> None:
    """Submit a single job via sbatch."""
    if dry_run:
        sbatch_script = build_sbatch_command(
            job_name, train_args, log_dir, redact_secrets=True
        )
        print(f"=== Job: {job_name} ===")
        print(sbatch_script)
        print()
        return

    sbatch_script = build_sbatch_command(job_name, train_args, log_dir)

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run DeepSVDD training jobs via SLURM")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting"
    )
    parser.add_argument(
        "--server",
        choices=list(SERVER_PATHS.keys()),
        default="cluster1",
        help="Server to use for paths (default: cluster1)",
    )
    parser.add_argument(
        "--env", required=True, choices=ENVS, help="Environment to train on"
    )
    parser.add_argument(
        "--prefix",
        help=(
            "Experiment group prefix for normal runs. Required unless selecting "
            "an optional preset with --variant."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_PRESETS.keys()),
        help=(
            "Optional DeepSVDD preset (currently: paper-regularized; see "
            "docs/image_svdd_collapse_bugs.md). Overrides --prefix and adds "
            "the preset's DeepSVDD CLI flags to train_svdd.py."
        ),
    )
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
    parser.add_argument(
        "--rollout-dir",
        help="Override rollout directory path, or pass a specific rollout .pt file",
    )
    parser.add_argument(
        "--rollouts-prefix",
        default=TRAIN_DEFAULTS["rollouts_prefix"],
        help=(
            "Rollout batch folder under the server rollouts base "
            f"(default: {TRAIN_DEFAULTS['rollouts_prefix']})."
        ),
    )
    parser.add_argument(
        "--rollout-max-levels",
        type=int,
        default=TRAIN_DEFAULTS["rollout_max_levels"],
        help=(
            "Maximum completed rollout levels to load from the selected rollout "
            "artifact. Omit to use the full largest artifact."
        ),
    )
    parser.add_argument(
        "--svdd-val-levels",
        type=int,
        default=TRAIN_DEFAULTS["svdd_val_levels"],
        help=(
            "Number of fixed validation levels to collect for the SVDD "
            "validation loss curve (default: 64)."
        ),
    )
    # Override checkpoints if needed
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help=(
            "Disable wandb logging for this submission. Pass when running on a "
            "cluster node without a configured wandb API key. Equivalent to "
            "dropping the -wandb flag and passing -wandb_mode disabled."
        ),
    )
    parser.add_argument(
        "--no-export-wandb-key",
        action="store_true",
        help=(
            "Do not embed the submitter's WANDB_API_KEY into the sbatch script. "
            "By default the key is read from $WANDB_API_KEY (preferred) or "
            "~/.netrc on the submitter, exported on the compute node for the "
            "duration of the job, and unset on exit via a trap. Use this flag "
            "if you don't want the key to appear in the job script (and your "
            "cluster's $HOME is shared with compute nodes, so ~/.netrc-based "
            "auth already works there)."
        ),
    )
    args = parser.parse_args()

    # Resolve --variant before --prefix is needed downstream.
    extra_python_args: list[str] = []
    if args.variant is not None:
        preset_prefix, preset_flags = VARIANT_PRESETS[args.variant]
        if args.prefix is not None and args.prefix != preset_prefix:
            print(
                f"Error: --variant {args.variant} already sets --prefix to "
                f"{preset_prefix}; passing --prefix {args.prefix} is "
                "inconsistent. Drop one or the other."
            )
            return 1
        args.prefix = preset_prefix
        extra_python_args = list(preset_flags)
    elif args.prefix is None:
        print("Error: either --prefix or --variant must be provided.")
        return 1

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    if args.svdd_val_levels <= 0:
        print(f"Error: --svdd-val-levels must be positive, got {args.svdd_val_levels}")
        return 1

    # Resolve the wandb API key once on the submitter and embed it in the
    # sbatch script(s) below. Skipped when wandb is disabled or the user opted
    # out of key export. A `trap`+`unset` in the sbatch script ensures the key
    # never leaks past the job process.
    wandb_api_key: Optional[str] = None
    if not args.no_wandb and not args.no_export_wandb_key:
        wandb_api_key = _get_wandb_api_key()
        if wandb_api_key is None:
            print(
                "Warning: --no-export-wandb-key not set, but no WANDB_API_KEY "
                "in env and no api.wandb.ai entry in ~/.netrc. The job will "
                "fall back to whatever wandb auth exists on the compute node."
            )

    # Get server-specific paths
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    rollouts_base_path = paths["rollouts_base"]
    seeds_base_path = paths["seeds_base"]
    svdd_output_dir = get_svdd_output_dir(paths["svdd_base"], args.prefix)
    log_dir = (
        Path(paths["log_base"]) / "svdd_train" / args.prefix / date.today().isoformat()
    )
    rollouts_prefix = args.rollouts_prefix
    svdd_env_name = get_svdd_env_name(args.env)

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {args.config}")
        print(f"Environment: {args.env}")
        print(f"Procgen env name: {svdd_env_name}")
        print(f"Feature type: {args.feature_type}")
        print(f"Prefix: {args.prefix}")
        print(f"Rollouts prefix: {rollouts_prefix}")
        print(f"SVDD output dir: {svdd_output_dir}")
        print(f"Log dir: {log_dir}")
        print(f"Experiment IDs: {args.exp_ids}")
        print(f"Rollout max levels: {args.rollout_max_levels}")
        print(f"SVDD validation levels: {args.svdd_val_levels}")
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
            rollout_dir = get_rollout_dir(
                args.env, exp_id, rollouts_base_path, rollouts_prefix
            )
        level_seeds_file = Path(seeds_base_path) / f"{exp_id}.json"

        # Get seed for this experiment ID
        seed = EXP_ID_TO_SEED.get(exp_id, exp_id)

        # Validate checkpoints exist
        missing = False
        for name, path in checkpoints.items():
            if not Path(path).exists():
                print(f"Warning: exp{exp_id} {name} checkpoint not found: {path}")
                missing = True

        # Check rollout directory exists
        resolved_rollout_path = resolve_rollout_path_for_check(rollout_dir)
        if not resolved_rollout_path.exists():
            print(
                f"Warning: exp{exp_id} rollout directory not found: "
                f"{resolved_rollout_path}"
            )
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
        feature_suffix = "latent" if args.feature_type == "hidden" else "image"
        job_name = f"svdd_{args.env}_{feature_suffix}_exp{exp_id}"
        wandb_group = f"{args.prefix}_{args.env}_{feature_suffix}_exp{exp_id}"
        expected_model_file = get_expected_model_file(svdd_output_dir, job_name)

        if args.dry_run:
            print(f"=== exp{exp_id} ===")
            print(f"  Job name: {job_name}")
            print(f"  Wandb group: {wandb_group}")
            print(f"  Feature type: {args.feature_type}")
            print(f"  Procgen env name: {svdd_env_name}")
            print(f"  Weak:   {checkpoints['weak']}")
            print(f"  Strong: {checkpoints['strong']}")
            print(f"  Rollout dir: {rollout_dir}")
            print(f"  Resolved rollout path: {resolved_rollout_path}")
            print(f"  Rollout max levels: {args.rollout_max_levels}")
            print(f"  Level seeds: {level_seeds_file}")
            print(f"  SVDD validation levels: {args.svdd_val_levels}")
            print(f"  Output dir: {svdd_output_dir}")
            print(f"  Expected model dir: {expected_model_file.parent}")
            print(f"  Expected model file: {expected_model_file}")
            print(f"  Log dir: {log_dir}")
            print(f"  Seed: {seed}")
            print()

        train_args = {
            "config": args.config,
            "name": job_name,
            "wandb_group": wandb_group,
            "env_name": svdd_env_name,
            "feature_type": args.feature_type,
            "cp_method": args.cp_method,
            "num_rollouts": args.num_rollouts,
            "query_cost": args.query_cost,
            "rollout_dir": rollout_dir,
            "rollout_max_levels": args.rollout_max_levels,
            "level_seeds_file": str(level_seeds_file),
            "svdd_val_levels": args.svdd_val_levels,
            "output_dir": str(svdd_output_dir),
            "seed": seed,
            "extra_python_args": extra_python_args,
            "use_wandb": not args.no_wandb,
            "wandb_api_key": wandb_api_key,
            **checkpoints,
        }

        submit_job(job_name, train_args, log_dir, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Script to run eval_strong_on_help.py jobs in parallel via SLURM sbatch.

This script finds existing evaluation NPZ files and re-evaluates
the strong agent on the seeds where help was requested.
"""

import argparse
import re
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from common import (
    ENVS,
    METHOD_CONFIGS,
    ROBUST_MAZE_CHECKPOINT_STEPS,
    SERVER_PATHS,
    find_newest_timestamp_dir,
    get_robust_maze_strong_checkpoint,
    get_strong_checkpoint,
    normalize_method_name,
)


ROBUST_SUFFIX_RE = re.compile(r"_(robust\d+)$")


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

DEFAULT_CONDA_ENV = "ood-stable"

SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",  # Shorter time since we're only re-evaluating
    "mem": "50G",
    "cpus-per-task": "16",
}

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def main():
    """Main function that orchestrates the entire workflow."""
    args = parse_arguments()
    if args.method is not None:
        args.method = normalize_method_name(args.method)

    # Initialize wandb if requested
    wandb_run = initialize_wandb(args) if args.use_wandb else None

    # Get paths based on server configuration
    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    evals_base = args.evals_base or paths["evals_base"]

    # Build log directory: <log_base>/strong_reval/<prefix>/<date>
    log_dir = (
        Path(paths["log_base"])
        / "strong_reval"
        / args.prefix
        / date.today().isoformat()
    )
    print(f"Log dir: {log_dir}")

    # Display configuration in dry run mode
    if args.dry_run:
        display_configuration(args, evals_base)

    # Find NPZ files to process
    npz_files = search_for_npz_files(args, evals_base)

    if not npz_files:
        print("\nNo NPZ files found matching criteria")
        if wandb_run:
            log_to_wandb({"npz_files_found": 0})
        return 1

    # Process all NPZ files
    stats = process_npz_files(args, npz_files, checkpoint_base_path, log_dir, wandb_run)

    # Display summary
    display_summary(args, npz_files, stats)

    # Finish wandb run
    if wandb_run:
        finalize_wandb(npz_files, stats)

    return 0 if stats["failed"] == 0 else 1


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================


def parse_arguments():
    """Parse command line arguments."""
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
        choices=list(SERVER_PATHS.keys()),
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
        help="Filter to specific method (e.g., max-prob). If not set, processes all methods found.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite experiment folder if it exists",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy/argmax strong-agent actions. Default is stochastic sampling to match scripts/run_eval.py.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging for job submission progress",
    )
    return parser.parse_args()


# ==============================================================================
# WANDB INTEGRATION
# ==============================================================================


def initialize_wandb(args):
    """Initialize wandb run if requested."""
    try:
        import wandb

        run_name = f"strong_reval_{args.env}_{args.prefix}"
        wandb.init(
            project="yrc-bench-strong-reval",
            name=run_name,
            config={
                "env": args.env,
                "prefix": args.prefix,
                "exp_ids": args.exp_ids,
                "method": args.method,
                "server": args.server,
                "qos": args.qos,
                "greedy": args.greedy,
            },
        )
        print(f"Initialized wandb run: {run_name}")
        return wandb
    except ImportError:
        print("Warning: wandb not available, disabling wandb logging")
        args.use_wandb = False
        return None


def log_to_wandb(data):
    """Log data to wandb if available."""
    if "wandb" in globals():
        globals()["wandb"].log(data)


def finalize_wandb(npz_files, stats):
    """Finalize wandb run with summary statistics."""
    if "wandb" in globals():
        wandb = globals()["wandb"]
        wandb.log(
            {
                "total_npz_files": len(npz_files),
                "jobs_submitted": stats["submitted"],
                "jobs_skipped": stats["skipped"],
                "jobs_failed": stats["failed"],
                "processing_time": stats["processing_time"],
                "success_rate": stats["submitted"] / len(npz_files)
                if len(npz_files) > 0
                else 0,
            }
        )
        wandb.finish()


# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================


def display_configuration(args, evals_base):
    """Display configuration information in dry run mode."""
    print(f"Server: {args.server}")
    print(f"Evals base: {evals_base}")
    print(f"Environment: {args.env}")
    print(f"Prefix: {args.prefix}")
    print(f"Method filter: {args.method or 'all'}")
    print(f"Greedy actions: {args.greedy}")
    print()


def display_summary(args, npz_files, stats):
    """Display final summary of job submissions."""
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"  Total NPZ files found: {len(npz_files)}")
    print(
        f"  Jobs {'would be ' if args.dry_run else ''}submitted: {stats['submitted']}"
    )
    print(f"  Skipped: {stats['skipped']}")
    if stats["failed"] > 0:
        print(f"  Failed: {stats['failed']}")
    print(f"  Processing time: {stats['processing_time']:.1f}s")
    print(f"{'=' * 60}")


# ==============================================================================
# NPZ FILE DISCOVERY
# ==============================================================================


def search_for_npz_files(args, evals_base):
    """Search for NPZ files matching the criteria."""
    print(f"\nSearching for NPZ files in {evals_base}...")
    print(f"  Environment: {args.env}")
    print(f"  Prefix: {args.prefix}")
    print(f"  Exp IDs: {args.exp_ids}")
    print(f"  Method filter: {args.method or 'all'}")

    start_search = time.time()
    npz_files = find_eval_npz_files(
        evals_base, args.prefix, args.env, args.exp_ids, args.method
    )
    search_time = time.time() - start_search

    print(
        f"\nFound {len(npz_files)} NPZ files to process (search took {search_time:.1f}s)"
    )

    if args.use_wandb:
        log_to_wandb({"npz_files_found": len(npz_files), "search_time": search_time})

    return npz_files


def find_eval_npz_files(
    evals_base: str,
    prefix: str,
    env: str,
    exp_ids: List[int],
    method_filter: Optional[str] = None,
) -> List[Tuple[Path, str, int, Optional[str]]]:
    """Find NPZ files from existing evaluations.

    Returns:
        List of tuples: (npz_path, method_name, exp_id, robust_key)
        where robust_key is e.g. "robust400" for robust runs, else None.
    """
    evals_path = Path(evals_base)
    results = []

    for exp_id in exp_ids:
        # Look for experiment group directories
        exp_group_pattern = f"{prefix}_{env}_exp{exp_id}"
        exp_group_dir = evals_path / exp_group_pattern

        if not exp_group_dir.exists():
            print(f"Warning: Experiment group dir not found: {exp_group_dir}")
            continue

        # Look for method subdirectories
        for method_dir in exp_group_dir.iterdir():
            if not method_dir.is_dir():
                continue

            # Extract method name from directory name
            match = re.match(rf"{env}_(.+)_exp{exp_id}$", method_dir.name)
            if not match:
                continue

            raw_method = match.group(1)
            robust_match = ROBUST_SUFFIX_RE.search(raw_method)
            if robust_match:
                robust_key = robust_match.group(1)
                raw_method = raw_method[: robust_match.start()]
            else:
                robust_key = None

            method_name = normalize_method_name(raw_method)

            # Apply method filter if specified
            if method_filter and method_name != method_filter:
                continue

            # Find the timestamp directory
            ts_dir = find_newest_timestamp_dir(method_dir, allow_compact_timestamp=True)
            if not ts_dir:
                print(f"Warning: No timestamp dir found in {method_dir}")
                continue

            # Look for NPZ files
            npz_files = list(ts_dir.glob("eval_seed_*_test.npz"))
            if not npz_files:
                print(f"Warning: No NPZ files found in {ts_dir}")
                continue

            for npz_file in npz_files:
                results.append((npz_file, method_name, exp_id, robust_key))

    return results


# ==============================================================================
# NPZ FILE PROCESSING
# ==============================================================================


def process_npz_files(args, npz_files, checkpoint_base_path, log_dir, wandb_run):
    """Process all NPZ files and submit jobs."""
    stats = {
        "submitted": 0,
        "skipped": 0,
        "failed": 0,
        "processing_time": 0.0,
    }

    print("\nProcessing NPZ files...")
    start_processing = time.time()

    for idx, (npz_path, method_name, exp_id, robust_key) in enumerate(npz_files, 1):
        label = f"{method_name} exp{exp_id}"
        if robust_key:
            label += f" ({robust_key})"
        print(f"\n[{idx}/{len(npz_files)}] Processing {label}...")

        if wandb_run:
            log_to_wandb(
                {
                    "processing_file": idx,
                    "total_files": len(npz_files),
                    "current_method": method_name,
                    "current_exp_id": exp_id,
                    "current_robust_key": robust_key,
                }
            )

        process_single_npz(
            args,
            npz_path,
            method_name,
            exp_id,
            robust_key,
            checkpoint_base_path,
            log_dir,
            stats,
            wandb_run,
        )

    stats["processing_time"] = time.time() - start_processing
    return stats


def process_single_npz(
    args,
    npz_path,
    method_name,
    exp_id,
    robust_key,
    checkpoint_base_path,
    log_dir,
    stats,
    wandb_run,
):
    """Process a single NPZ file."""
    # Get strong checkpoint
    strong_path = get_strong_checkpoint_path(
        args, exp_id, robust_key, checkpoint_base_path
    )
    if not validate_strong_checkpoint(strong_path, exp_id, stats, wandb_run):
        return

    # Get config path
    config_path = get_config_path(args, method_name, npz_path, stats, wandb_run)
    if not config_path:
        return

    # Check if output already exists
    strong_output_path, full_budget_path = output_paths(npz_path)
    if should_skip_existing(
        strong_output_path,
        full_budget_path,
        args,
        npz_path,
        stats,
        wandb_run,
    ):
        return

    # Build job name and submit
    job_method_label = f"{method_name}_{robust_key}" if robust_key else method_name
    job_name = f"strong_reval_{args.env}_{job_method_label}_exp{exp_id}"

    if args.dry_run:
        display_job_info(
            job_name,
            npz_path,
            strong_path,
            config_path,
            strong_output_path,
            full_budget_path,
            args,
        )
        stats["submitted"] += 1
    else:
        submit_job_wrapper(
            job_name,
            npz_path,
            strong_path,
            config_path,
            args,
            log_dir,
            stats,
            wandb_run,
        )


def get_strong_checkpoint_path(args, exp_id, robust_key, checkpoint_base_path):
    """Get the strong checkpoint path."""
    if args.strong:
        return args.strong
    if robust_key is not None:
        if args.env != "maze":
            print(
                f"  ⚠️  Warning: robust suffix '{robust_key}' on a {args.env!r} run is unexpected "
                f"(robust checkpoints are maze-only); falling back to non-robust strong."
            )
        else:
            return get_robust_maze_strong_checkpoint(
                exp_id,
                checkpoint_base_path,
                ROBUST_MAZE_CHECKPOINT_STEPS[robust_key],
            )
    return get_strong_checkpoint(
        args.env,
        exp_id,
        checkpoint_base_path,
        allow_compact_timestamp=True,
    )


def validate_strong_checkpoint(strong_path, exp_id, stats, wandb_run):
    """Validate that the strong checkpoint exists."""
    if not strong_path:
        print(f"  ⚠️  Warning: Strong checkpoint not found for exp{exp_id}, skipping")
        stats["skipped"] += 1
        if wandb_run:
            log_to_wandb(
                {
                    "skip_reason": "strong_checkpoint_not_found",
                    "skipped_total": stats["skipped"],
                }
            )
        return False

    if not Path(strong_path).exists():
        print(
            f"  ⚠️  Warning: Strong checkpoint does not exist: {strong_path}, skipping"
        )
        stats["skipped"] += 1
        if wandb_run:
            log_to_wandb(
                {
                    "skip_reason": "strong_checkpoint_missing",
                    "skipped_total": stats["skipped"],
                }
            )
        return False

    return True


def get_config_path(args, method_name, npz_path, stats, wandb_run):
    """Determine the config path for the method."""
    if args.config:
        effective_config = args.config
    elif method_name in METHOD_CONFIGS:
        effective_config = f"configs/eval/{args.env}/{METHOD_CONFIGS[method_name]}"
    else:
        print(f"  ⚠️  Warning: Unknown method '{method_name}' for {npz_path}, skipping")
        stats["skipped"] += 1
        if wandb_run:
            log_to_wandb(
                {"skip_reason": "unknown_method", "skipped_total": stats["skipped"]}
            )
        return None

    if not Path(effective_config).exists():
        print(f"  ⚠️  Warning: Config not found: {effective_config}, skipping")
        stats["skipped"] += 1
        if wandb_run:
            log_to_wandb(
                {"skip_reason": "config_not_found", "skipped_total": stats["skipped"]}
            )
        return None

    return effective_config


def output_paths(npz_path):
    strong_output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
    full_budget_path = npz_path.with_name(f"{npz_path.stem}_full_budget_eval.npz")
    return strong_output_path, full_budget_path


def should_skip_existing(
    strong_output_path,
    full_budget_path,
    args,
    npz_path,
    stats,
    wandb_run,
):
    """Check if output already exists and should be skipped."""
    if strong_output_path.exists() and full_budget_path.exists() and not args.overwrite:
        print(
            f"  ⏭️  Skipping {npz_path.name} - outputs already exist: "
            f"{strong_output_path}, {full_budget_path}"
        )
        stats["skipped"] += 1
        if wandb_run:
            log_to_wandb(
                {"skip_reason": "output_exists", "skipped_total": stats["skipped"]}
            )
        return True
    return False


def display_job_info(
    job_name,
    npz_path,
    strong_path,
    config_path,
    strong_output_path,
    full_budget_path,
    args,
):
    """Display job information in dry run mode."""
    print(f"=== {job_name} ===")
    print(f"  NPZ file: {npz_path}")
    print(f"  Strong:   {strong_path}")
    print(f"  Config:   {config_path}")
    print(f"  Strong reval output: {strong_output_path}")
    print(f"  Full-budget output:  {full_budget_path}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Greedy:    {args.greedy}")
    print()


def submit_job_wrapper(
    job_name, npz_path, strong_path, config_path, args, log_dir, stats, wandb_run
):
    """Wrapper to submit job and handle results."""
    try:
        submit_job(
            job_name,
            str(npz_path),
            strong_path,
            config_path,
            args.conda_env,
            log_dir,
            args.qos,
            greedy=args.greedy,
            overwrite=args.overwrite,
            dry_run=False,
        )
        stats["submitted"] += 1
        print(f"  ✓ Successfully submitted job: {job_name}")

        if wandb_run:
            log_to_wandb(
                {
                    "submitted_total": stats["submitted"],
                    "job_name": job_name,
                    "submission_success": True,
                }
            )
    except Exception as e:
        stats["failed"] += 1
        print(f"  ❌ Failed to submit job: {e}")

        if wandb_run:
            log_to_wandb(
                {
                    "failed_total": stats["failed"],
                    "job_name": job_name,
                    "submission_success": False,
                    "error": str(e),
                }
            )


# ==============================================================================
# JOB SUBMISSION
# ==============================================================================


def submit_job(
    job_name: str,
    npz_file: str,
    strong_path: str,
    config_path: str,
    conda_env: str,
    log_dir: Path,
    qos: str = "default",
    greedy: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Submit a single job via sbatch."""
    sbatch_script = build_sbatch_script(
        job_name,
        npz_file,
        strong_path,
        config_path,
        conda_env,
        log_dir,
        qos,
        greedy,
        overwrite,
    )

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
        # Extract job ID from sbatch output
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        job_id = job_id_match.group(1) if job_id_match else "unknown"
        print(f"  → Submitted {job_name} (Job ID: {job_id})")
    else:
        print(f"  → Failed to submit {job_name}: {result.stderr}")
        raise RuntimeError(f"sbatch failed: {result.stderr}")


def build_sbatch_script(
    job_name: str,
    npz_file: str,
    strong_path: str,
    config_path: str,
    conda_env: str,
    log_dir: Path,
    qos: str = "default",
    greedy: bool = False,
    overwrite: bool = False,
) -> str:
    """Build the sbatch script for job submission."""
    # Override QOS in SLURM config
    slurm_config = SLURM_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())

    # Build the python command
    python_cmd = (
        f"python eval_strong_on_help.py -c {config_path} -n {job_name} "
        f"--npz_file {npz_file} -strong {strong_path} "
        f"-greedy {str(greedy).lower()}"
    )
    if overwrite:
        python_cmd += " --overwrite"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""
    return sbatch_script


# ==============================================================================
# PATH UTILITIES
# ==============================================================================

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    exit(main())

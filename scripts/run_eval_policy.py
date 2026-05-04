#!/usr/bin/env python3
"""
Submit eval_policy.py jobs via SLURM.

This mirrors scripts/run_eval.py for standalone acting-policy evaluation:
- uses the same checkpoint lookup convention
- uses the same fixed level seed files as run_eval.py
- evaluates sim / weak / strong agent checkpoints directly
"""

import argparse
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict, List

from common import (
    ENVS,
    ROBUST_MAZE_CHECKPOINT_STEPS,
    SERVER_PATHS,
    get_checkpoints,
    get_eval_env_name,
    get_robust_maze_strong_checkpoint,
)


DEFAULT_CONDA_ENV = "ood-stable"

SLURM_CONFIG = {
    "qos": "default",
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "mem": "50G",
    "cpus-per-task": "16",
}

POLICIES = ["sim", "weak", "strong"]

DEFAULT_CONFIGS = {
    "coinrun": "configs/eval/coinrun/max_prob.yaml",
    "coinrun_proxy_fail": "configs/eval/coinrun/max_prob.yaml",
    "maze": "configs/eval/maze/max_prob.yaml",
}

EVAL_ENVS = [*ENVS, "coinrun_proxy_fail"]

CHECKPOINT_ENVS = {
    "coinrun_proxy_fail": "coinrun",
}

EVAL_DEFAULTS = {
    "num_rollouts": 5000,
    "eval_split": "test",
    "video_logging_mode": "none",
    "wandb_mode": "disabled",
    "greedy": False,
}


def build_sbatch_command(
    job_name: str,
    eval_args: Dict[str, object],
    conda_env: str,
    log_dir: Path,
    qos: str = "default",
) -> str:
    """Build the sbatch submission script."""
    slurm_config = SLURM_CONFIG.copy()
    slurm_config["qos"] = qos
    slurm_args = " ".join(f"--{k}={v}" for k, v in slurm_config.items())

    python_args: List[str] = [
        "python eval_policy.py",
        f"-c {eval_args['config']}",
        f"-n {eval_args['name']}",
        f"-experiment_group {eval_args['experiment_group']}",
        f"-eval_split {eval_args['eval_split']}",
        f"-en {eval_args['env_name']}",
        f"--model_file {eval_args['model_file']}",
        f"-num_rollouts {eval_args['num_rollouts']}",
        f"-level_seeds_file {eval_args['level_seeds_file']}",
        f"-video_logging_mode {eval_args['video_logging_mode']}",
        f"-wandb_mode {eval_args['wandb_mode']}",
        f"-greedy {str(eval_args['greedy']).lower()}",
    ]

    if eval_args.get("wandb_project"):
        python_args.append(f"-wandb_project {eval_args['wandb_project']}")
    if eval_args.get("random_percent") is not None:
        python_args.append(f"-random_percent {eval_args['random_percent']}")
    if eval_args.get("num_envs") is not None:
        python_args.append(f"-num_envs {eval_args['num_envs']}")
    if eval_args.get("max_steps") is not None:
        python_args.append(f"-max_steps {eval_args['max_steps']}")
    if eval_args.get("device") is not None:
        python_args.append(f"-d {eval_args['device']}")

    python_cmd = " ".join(python_args)

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{chr(10).join(f"#SBATCH --{k}={v}" for k, v in slurm_config.items())}

echo "Using conda env: {conda_env}"
eval "$(conda shell.bash hook)"
conda activate {conda_env}
srun {slurm_args} {python_cmd}
"""


def submit_job(
    job_name: str,
    eval_args: Dict[str, object],
    conda_env: str,
    log_dir: Path,
    qos: str = "default",
    dry_run: bool = False,
) -> None:
    """Submit or print a single job."""
    sbatch_script = build_sbatch_command(job_name, eval_args, conda_env, log_dir, qos)

    if dry_run:
        print(f"=== Job: {job_name} ===")
        print(sbatch_script)
        print()
        return

    log_dir.mkdir(parents=True, exist_ok=True)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eval_policy.py jobs via SLURM using the same seeds as run_eval.py"
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
        "--env", required=True, choices=EVAL_ENVS, help="Environment to evaluate"
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
        "--agents",
        nargs="+",
        choices=POLICIES,
        default=["strong"],
        help="Acting policies to evaluate (default: strong)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override config file path. Defaults to the env's max_prob eval config.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=EVAL_DEFAULTS["num_rollouts"],
        help=f"Number of episodes to evaluate (default: {EVAL_DEFAULTS['num_rollouts']})",
    )
    parser.add_argument(
        "--eval-split",
        default=EVAL_DEFAULTS["eval_split"],
        choices=["train", "val_sim", "val_true", "test"],
        help=f"Environment split to evaluate (default: {EVAL_DEFAULTS['eval_split']})",
    )
    parser.add_argument(
        "--random-percent",
        type=int,
        default=None,
        help="Override environment.test.random_percent (e.g. 0 for ID-only, 100 for OOD-only)",
    )
    parser.add_argument(
        "--video-logging-mode",
        default=EVAL_DEFAULTS["video_logging_mode"],
        choices=["wandb", "folder", "both", "none"],
        help=f"Video logging mode (default: {EVAL_DEFAULTS['video_logging_mode']})",
    )
    parser.add_argument(
        "--wandb-mode",
        default=EVAL_DEFAULTS["wandb_mode"],
        choices=["online", "offline", "disabled"],
        help=f"WandB mode (default: {EVAL_DEFAULTS['wandb_mode']})",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=EVAL_DEFAULTS["greedy"],
        help="Use greedy/argmax action selection. Default is stochastic sampling to match run_eval.py.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override wandb project name",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel environments",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum number of steps per episode",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Override device id (-1 for CPU)",
    )
    parser.add_argument("--sim", help="Override sim weak checkpoint path")
    parser.add_argument("--weak", help="Override weak checkpoint path")
    parser.add_argument("--strong", help="Override strong checkpoint path")
    robust_group = parser.add_mutually_exclusive_group()
    robust_group.add_argument(
        "--robust200",
        "--robust-200",
        action="store_true",
        help=(
            "For maze strong-policy evals, use the random-start strong policy "
            "at 200M timesteps."
        ),
    )
    robust_group.add_argument(
        "--robust400",
        "--robust-400",
        action="store_true",
        help=(
            "For maze strong-policy evals, use the random-start strong policy "
            "at 400M timesteps."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    paths = SERVER_PATHS[args.server]
    checkpoint_base_path = paths["checkpoint_base"]
    seeds_base_path = paths["seeds_base"]
    checkpoint_env = CHECKPOINT_ENVS.get(args.env, args.env)
    eval_env_name = get_eval_env_name(args.env)
    robust_checkpoint_key = None
    if args.robust200:
        robust_checkpoint_key = "robust200"
    elif args.robust400:
        robust_checkpoint_key = "robust400"

    if robust_checkpoint_key is not None and checkpoint_env != "maze":
        print(
            f"Error: --{robust_checkpoint_key} is currently supported only for maze."
        )
        return 1
    if robust_checkpoint_key is not None and args.strong:
        print(f"Error: pass either --strong or --{robust_checkpoint_key}, not both.")
        return 1
    if robust_checkpoint_key is not None and args.agents != ["strong"]:
        print(f"Error: --{robust_checkpoint_key} can only be used with --agents strong.")
        return 1
    output_prefix = (
        f"{args.prefix}_{robust_checkpoint_key}"
        if robust_checkpoint_key is not None
        else args.prefix
    )

    config_path = args.config or DEFAULT_CONFIGS[args.env]
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Conda env: {args.conda_env}")

    log_base = Path(paths["log_base"])
    wandb_project = args.wandb_project or "policy_eval"
    log_dir = log_base / wandb_project / output_prefix / date.today().isoformat()
    print(f"Log dir: {log_dir}")

    if args.dry_run:
        print(f"Server: {args.server}")
        print(f"Config: {config_path}")
        print(f"Environment: {args.env}")
        if eval_env_name != args.env:
            print(f"Procgen env: {eval_env_name}")
        print(f"Agents: {' '.join(args.agents)}")
        print(f"Greedy actions: {args.greedy}")
        print(f"Prefix: {args.prefix}")
        if output_prefix != args.prefix:
            print(f"Output prefix: {output_prefix}")
        if robust_checkpoint_key is not None:
            print(
                "Robust strong checkpoint: "
                f"{robust_checkpoint_key} "
                f"({ROBUST_MAZE_CHECKPOINT_STEPS[robust_checkpoint_key]} timesteps)"
            )
        print()

    if args.dry_run and checkpoint_env != args.env:
        print(f"Checkpoint env: {checkpoint_env}")
        print()

    for exp_id in args.exp_ids:
        checkpoints = get_checkpoints(checkpoint_env, exp_id, checkpoint_base_path)
        if args.sim:
            checkpoints["sim"] = args.sim
        if args.weak:
            checkpoints["weak"] = args.weak
        if args.strong:
            checkpoints["strong"] = args.strong
        elif robust_checkpoint_key is not None:
            checkpoints["strong"] = get_robust_maze_strong_checkpoint(
                exp_id,
                checkpoint_base_path,
                ROBUST_MAZE_CHECKPOINT_STEPS[robust_checkpoint_key],
            )

        level_seeds_file = Path(seeds_base_path) / f"{exp_id}.json"
        if not level_seeds_file.exists():
            print(
                f"Warning: exp{exp_id} level seeds file not found: {level_seeds_file}"
            )
            print(f"Skipping exp{exp_id}\n")
            continue

        for agent in args.agents:
            model_file = checkpoints[agent]
            if not Path(model_file).exists():
                print(
                    f"Warning: exp{exp_id} {agent} checkpoint not found: {model_file}"
                )
                print(f"Skipping exp{exp_id} {agent}\n")
                continue

            name_parts = [args.env, agent, "policy"]
            if agent == "strong" and robust_checkpoint_key is not None:
                name_parts.append(robust_checkpoint_key)
            name_parts.append(f"exp{exp_id}")
            job_name = "_".join(name_parts)
            experiment_group = f"{output_prefix}_{args.env}_{agent}_exp{exp_id}"

            if args.dry_run:
                print(f"=== exp{exp_id} / {agent} ===")
                print(f"  Job name: {job_name}")
                print(f"  Experiment group: {experiment_group}")
                print(f"  Model:  {model_file}")
                print(f"  Seeds:  {level_seeds_file}")
                print()
                continue

            eval_args: Dict[str, object] = {
                "config": config_path,
                "name": job_name,
                "experiment_group": experiment_group,
                "env_name": eval_env_name,
                "model_file": model_file,
                "level_seeds_file": str(level_seeds_file),
                "num_rollouts": args.num_rollouts,
                "eval_split": args.eval_split,
                "video_logging_mode": args.video_logging_mode,
                "wandb_mode": args.wandb_mode,
                "greedy": args.greedy,
                "wandb_project": args.wandb_project,
                "random_percent": args.random_percent,
                "num_envs": args.num_envs,
                "max_steps": args.max_steps,
                "device": args.device,
            }

            submit_job(
                job_name,
                eval_args,
                args.conda_env,
                log_dir,
                args.qos,
                dry_run=False,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

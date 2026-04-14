from pathlib import Path
from typing import Any, Optional

import numpy as np
import wandb


def init_eval_wandb_run(
    config,
    *,
    name: str,
    job_type: str,
    run_config: Any,
    project_fallback: Optional[str] = None,
    tolerate_failure: bool = False,
):
    """Initialize a wandb run for evaluation-style scripts."""
    project = config.wandb.project or project_fallback
    wandb_kwargs = {
        "name": name,
        "project": project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "job_type": job_type,
        "config": run_config,
    }

    if config.wandb.entity is not None:
        wandb_kwargs["entity"] = config.wandb.entity

    try:
        run = wandb.init(**wandb_kwargs)
    except Exception as exc:
        if tolerate_failure:
            print(f"Failed to initialize wandb: {exc}")
            return None
        raise

    print(f"\nInitialized wandb run: {run.name}")
    return run


def save_npz_results(output_path: Path, *, announce: bool = False, **arrays) -> None:
    """Save an evaluation artifact as an NPZ file."""
    np.savez(output_path, **arrays)
    if announce:
        print(f"\nResults saved to {output_path}")

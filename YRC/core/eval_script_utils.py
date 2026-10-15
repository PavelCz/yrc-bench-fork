from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

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
    """Atomically save an evaluation artifact as an NPZ file."""
    if not str(output_path).endswith(".npz"):
        output_path = Path(f"{output_path}.npz")
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("xb") as temporary_file:
            np.savez(temporary_file, **arrays)
            temporary_file.flush()
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if announce:
        print(f"\nResults saved to {output_path}")


class IntermediateEvaluationResults:
    """Checkpoint completed AFHP evaluations in the standard NPZ schema."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.afhps = []
        self.performances = []
        self.desired_percentiles = []
        self.meta = []
        self.order = []

    def record_evaluation(
        self,
        desired_percentile: Optional[float],
        afhp: float,
        performance: float,
        meta: Any,
    ) -> None:
        """Save one completed rollout while retaining prior checkpoint points."""
        self.afhps.append(float(afhp))
        self.performances.append(float(performance))
        self.desired_percentiles.append(
            float("nan") if desired_percentile is None else float(desired_percentile)
        )
        self.meta.append(meta)
        self.order.append(len(self.order) + 1)

        sort_order = np.argsort(self.afhps, kind="stable")
        ordered_meta = np.asarray(self.meta, dtype=object)[sort_order]
        save_npz_results(
            self.output_path,
            afhps=np.asarray(self.afhps, dtype=float)[sort_order],
            performances=np.asarray(self.performances, dtype=float)[sort_order],
            desired_percentiles=np.asarray(self.desired_percentiles, dtype=float)[
                sort_order
            ],
            meta=ordered_meta,
            order=np.asarray(self.order, dtype=int)[sort_order],
            sampling_info=np.array(
                [
                    {
                        "status": "in_progress",
                        "completed_evaluations": len(self.order),
                    }
                ],
                dtype=object,
            ),
        )

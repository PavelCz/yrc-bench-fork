"""
Parallel bin-based AFHP evaluation for YRC policies.

The AFHP space [0, 1] is divided into num_bins equal-width bins. Each bin is
assigned to a worker thread that finds a threshold achieving an AFHP in that bin.
Workers use the policy's calibrated train_percentile method as an initial heuristic,
then refine via binary search within the bin's percentile bracket.

Per-bin .npz checkpoint files allow restarting only failed bins.
"""

import copy
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from YRC.core import Evaluator
from YRC.policies.base import LevelBasedRandomPolicy, TimestepRandomPolicy
from YRC.policies.heuristic import ExponentialHeuristicPolicy, WaitPolicy
from YRC.policies.lightning_ae import LightningAEPolicy
from YRC.policies.ood import OODPolicy
from YRC.policies.threshold import ThresholdPolicy

try:
    import wandb
except ImportError:
    wandb = None


class EvalStepTracker:
    """Tracks evaluation steps and logs metrics to wandb."""

    def __init__(self, wandb_run: Optional[Any] = None):
        self.step = 0
        self.wandb_run = wandb_run

    def log_eval(
        self,
        threshold: float,
        step_afhp: float,
        level_afhp: float,
        performance: float,
    ):
        """Log evaluation metrics to console and wandb."""
        self.step += 1

        print(
            f"[Eval {self.step:3d}] threshold={threshold:10.4f}, "
            f"step_afhp={step_afhp:6.2f}%, level_afhp={level_afhp:6.2f}%, "
            f"performance={performance:.4f}"
        )

        if self.wandb_run is not None and wandb is not None:
            wandb.log(
                {
                    "eval/step": self.step,
                    "eval/threshold": (
                        threshold
                        if not np.isinf(threshold)
                        else (1e10 if threshold > 0 else -1e10)
                    ),
                    "eval/step_afhp": step_afhp,
                    "eval/level_afhp": level_afhp,
                    "eval/performance": performance,
                },
                step=self.step,
            )


def _eval_at_threshold(
    policy,
    threshold: float,
    evaluator: Evaluator,
    envs_factory,
    split: str,
    afhp_metric: str,
) -> Tuple[float, float, Dict[str, Any]]:
    """Evaluate policy at a threshold, returning (afhp, performance, meta).

    Args:
        policy: Coordination policy (threshold will be set via update_policy_params).
        threshold: Threshold value to evaluate at.
        evaluator: Evaluator for running episodes.
        envs_factory: Callable returning fresh environments.
        split: Evaluation split ("train", "val", "test").
        afhp_metric: Primary metric to return as afhp
            ("level_afhp" or "step_afhp").

    Returns:
        Tuple of (afhp [0,1], performance, meta dict).
    """
    update_policy_params(policy, threshold)
    envs = envs_factory()
    summary = evaluator.eval(policy, envs, [split], close_envs=True)

    level_ood_preds = summary[split]["level_ood_pred"]
    level_afhp = float(np.mean(level_ood_preds))
    step_afhp = float(summary[split]["action_1_frac"])
    performance = float(summary[split]["env_return_mean"])

    afhp = level_afhp if afhp_metric == "level_afhp" else step_afhp
    meta = {"summary": summary, "threshold": threshold}
    return afhp, performance, meta


def run_bin(
    bin_idx: int,
    bin_lo: float,
    bin_hi: float,
    policy,
    evaluator: Evaluator,
    envs_factory,
    split: str,
    afhp_metric: str,
    search_depth_limit: int,
) -> Tuple[float, float, Dict[str, Any], float]:
    """Find a threshold achieving AFHP in [bin_lo, bin_hi].

    Starts from the method-specific heuristic (train_percentile) as an initial
    guess, then refines via binary search within the percentile bracket for this bin.

    Higher percentile → higher threshold → lower AFHP (fewer help requests).
    The percentile bracket for bin i is [100 - bin_hi*100, 100 - bin_lo*100].

    Returns:
        (afhp, performance, meta, threshold) — result of the best evaluation found.
    """
    target_afhp = (bin_lo + bin_hi) / 2.0

    # Initial guess: convert target AFHP to percentile and query the heuristic
    percentile = 100.0 - target_afhp * 100.0
    if afhp_metric == "level_afhp":
        threshold = policy.train_percentile_level(percentile)
    else:
        threshold = policy.train_percentile_step(percentile)

    afhp, performance, meta = _eval_at_threshold(
        policy, threshold, evaluator, envs_factory, split, afhp_metric
    )
    print(
        f"[Bin {bin_idx}] Initial: target={target_afhp:.3f}, "
        f"got={afhp:.3f}, threshold={threshold:.4f}"
    )

    # Binary search within the percentile bracket for this bin
    lo_pct = 100.0 - bin_hi * 100.0  # low percentile → high AFHP
    hi_pct = 100.0 - bin_lo * 100.0  # high percentile → low AFHP

    for depth in range(search_depth_limit):
        if bin_lo <= afhp <= bin_hi:
            break

        mid_pct = (lo_pct + hi_pct) / 2.0
        if afhp_metric == "level_afhp":
            threshold = policy.train_percentile_level(mid_pct)
        else:
            threshold = policy.train_percentile_step(mid_pct)

        afhp, performance, meta = _eval_at_threshold(
            policy, threshold, evaluator, envs_factory, split, afhp_metric
        )
        print(
            f"[Bin {bin_idx}] Depth {depth + 1}: got={afhp:.3f}, "
            f"threshold={threshold:.4f}"
        )

        if afhp > bin_hi:
            # AFHP too high → need higher threshold → raise lo_pct
            lo_pct = mid_pct
        elif afhp < bin_lo:
            # AFHP too low → need lower threshold → lower hi_pct
            hi_pct = mid_pct

    if not (bin_lo <= afhp <= bin_hi):
        print(
            f"[Bin {bin_idx}] Warning: could not reach [{bin_lo:.3f}, {bin_hi:.3f}], "
            f"best achieved={afhp:.3f}"
        )

    return afhp, performance, meta, threshold


def run_parallel_eval(
    policy,
    evaluator: Evaluator,
    envs_factory,
    split: str,
    num_bins: int,
    results_path: Path,
    wandb_run=None,
    logger=None,
    afhp_metric: str = "level_afhp",
    max_workers: int = 1,
    search_depth_limit: int = 10,
) -> List[Dict[str, Any]]:
    """Run bin-based AFHP evaluation, optionally in parallel.

    Divides [0, 1] AFHP space into num_bins equal-width bins. Each bin is assigned
    to a worker that finds a threshold achieving AFHP in that bin. Workers use
    train_percentile_level/step as an initial heuristic, then refine via binary
    search within the bin's percentile bracket.

    Per-bin .npz checkpoints enable restarting only failed bins: if a checkpoint
    already exists for a bin, that bin is skipped.

    Args:
        policy: Calibrated policy with train_percentile_level/step methods.
        evaluator: Evaluator instance.
        envs_factory: Callable returning fresh environments.
        split: Evaluation split ("train", "val", "test").
        num_bins: Number of equal-width AFHP bins to fill.
        results_path: Base path for per-bin checkpoints. Checkpoints are saved as
            {stem}_bin_{i}.npz alongside the results file.
        wandb_run: Optional wandb run for logging.
        logger: Optional logger (currently unused, reserved for future use).
        afhp_metric: Which AFHP metric to target ("level_afhp" or "step_afhp").
        max_workers: Number of parallel worker threads. Default 1 = sequential.
            Increase with caution: workers share the evaluator and envs_factory,
            which may not be thread-safe for all environment backends.
        search_depth_limit: Max binary search iterations per bin.

    Returns:
        List of result dicts sorted by afhp, each with keys:
        afhp, performance, desired_percentile, meta, order.
    """
    results_path = Path(results_path)
    tracker = EvalStepTracker(wandb_run=wandb_run)

    # Derive base stem for checkpoint filenames (strip .npz if present)
    stem = results_path.stem if results_path.suffix == ".npz" else results_path.name

    # Build bin specs and checkpoint paths
    bin_specs: List[Tuple[int, float, float, Path]] = []
    for i in range(num_bins):
        bin_lo = i / num_bins
        bin_hi = (i + 1) / num_bins
        ckpt = results_path.parent / f"{stem}_bin_{i}.npz"
        bin_specs.append((i, bin_lo, bin_hi, ckpt))

    pending = [(i, lo, hi, cp) for i, lo, hi, cp in bin_specs if not cp.exists()]
    n_cached = len(bin_specs) - len(pending)
    print(
        f"Bins: {num_bins} total, {n_cached} cached, {len(pending)} to evaluate"
    )

    def run_one_bin(
        bin_idx: int, bin_lo: float, bin_hi: float, checkpoint_path: Path
    ) -> Tuple[int, float, float]:
        # Deep-copy policy so each worker has independent mutable state
        policy_copy = copy.deepcopy(policy)

        afhp, performance, meta, threshold = run_bin(
            bin_idx=bin_idx,
            bin_lo=bin_lo,
            bin_hi=bin_hi,
            policy=policy_copy,
            evaluator=evaluator,
            envs_factory=envs_factory,
            split=split,
            afhp_metric=afhp_metric,
            search_depth_limit=search_depth_limit,
        )

        np.savez(
            checkpoint_path,
            afhp=np.array([afhp]),
            performance=np.array([performance]),
            desired_percentile=np.array([(bin_lo + bin_hi) / 2.0]),
            meta=np.array([meta], dtype=object),
            threshold=np.array([threshold]),
        )

        # Log both metrics from the summary
        summary = meta["summary"]
        level_afhp_pct = float(np.mean(summary[split]["level_ood_pred"])) * 100.0
        step_afhp_pct = float(summary[split]["action_1_frac"]) * 100.0
        tracker.log_eval(
            threshold=threshold,
            step_afhp=step_afhp_pct,
            level_afhp=level_afhp_pct,
            performance=performance,
        )

        return bin_idx, afhp, performance

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_one_bin, i, lo, hi, cp): i
                for i, lo, hi, cp in pending
            }
            for future in as_completed(futures):
                bin_idx = futures[future]
                try:
                    i, afhp, perf = future.result()
                    print(f"[Bin {i}] Saved: afhp={afhp:.3f}, perf={perf:.4f}")
                except Exception:
                    print(f"[Bin {bin_idx}] Failed:")
                    traceback.print_exc()

    # Aggregate results from per-bin checkpoints
    results: List[Dict[str, Any]] = []
    for i, _bin_lo, _bin_hi, ckpt in bin_specs:
        if not ckpt.exists():
            print(f"Warning: Bin {i} checkpoint not found, skipping.")
            continue
        data = np.load(ckpt, allow_pickle=True)
        results.append(
            {
                "afhp": float(data["afhp"][0]),
                "performance": float(data["performance"][0]),
                "desired_percentile": float(data["desired_percentile"][0]),
                "meta": data["meta"][0],
                "order": i,
            }
        )

    return sorted(results, key=lambda r: r["afhp"])


def save_calibration_state(policy, path: Path) -> None:
    """Save policy calibration state to a .npz file for use by bin workers.

    Each policy type stores different calibration data:
    - ThresholdPolicy / OODPolicy / LightningAEPolicy: per-step and per-episode-max scores
    - TimestepRandomPolicy / ExponentialHeuristicPolicy: mean episode length
    - WaitPolicy: full episode length distribution

    Args:
        policy: Calibrated policy whose internal state to save.
        path: Destination path (numpy adds .npz if not present).

    Raises:
        ValueError: If no calibration state is found on the policy.
    """
    state: Dict[str, Any] = {}

    if hasattr(policy, "_train_scores") and policy._train_scores is not None:
        state["train_scores"] = policy._train_scores
    if (
        hasattr(policy, "_train_episode_max_scores")
        and policy._train_episode_max_scores is not None
    ):
        state["train_episode_max_scores"] = policy._train_episode_max_scores
    if (
        hasattr(policy, "_mean_episode_length")
        and policy._mean_episode_length is not None
    ):
        state["mean_episode_length"] = np.array([policy._mean_episode_length])
    if hasattr(policy, "_episode_lengths") and policy._episode_lengths is not None:
        state["episode_lengths"] = policy._episode_lengths

    if not state:
        raise ValueError(
            f"No calibration state found on policy {type(policy).__name__}. "
            "Run calibrate_percentile_mapping() before saving."
        )

    np.savez(path, **state)


def load_calibration_state(policy, path: Path) -> None:
    """Load calibration state from a .npz file into a policy.

    Injects whichever fields are present in the file into the corresponding
    policy attributes. The policy must already be initialised (model weights
    loaded) — only the calibration arrays are overwritten.

    Args:
        policy: Policy object to inject calibration state into.
        path: Path to the .npz file saved by save_calibration_state().
    """
    data = np.load(path)

    if "train_scores" in data and hasattr(policy, "_train_scores"):
        policy._train_scores = data["train_scores"]
    if "train_episode_max_scores" in data and hasattr(
        policy, "_train_episode_max_scores"
    ):
        policy._train_episode_max_scores = data["train_episode_max_scores"]
    if "mean_episode_length" in data and hasattr(policy, "_mean_episode_length"):
        policy._mean_episode_length = float(data["mean_episode_length"][0])
    if "episode_lengths" in data and hasattr(policy, "_episode_lengths"):
        policy._episode_lengths = data["episode_lengths"]


def update_policy_params(policy, threshold):
    if (
        isinstance(policy, LightningAEPolicy)
        or isinstance(policy, OODPolicy)
        or isinstance(policy, ThresholdPolicy)
    ):
        params = policy.params.copy()
        params["threshold"] = threshold
        policy.update_params(params)
    elif (
        isinstance(policy, TimestepRandomPolicy)
        or isinstance(policy, LevelBasedRandomPolicy)
        or isinstance(policy, ExponentialHeuristicPolicy)
    ):
        if threshold == float("inf"):
            threshold = 0.0
        elif threshold == float("-inf"):
            threshold = 1.0
        policy.update_params(threshold)
    elif isinstance(policy, WaitPolicy):
        if threshold == float("inf"):
            threshold = 10000
        elif threshold == float("-inf"):
            threshold = 0
        policy.update_params(threshold=threshold)
    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )

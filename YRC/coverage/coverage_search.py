"""
Parallel bin-based AFHP evaluation for YRC policies.

The AFHP space [0, 1] is divided into num_bins equal-width bins. Each bin is
assigned to a worker thread that finds a threshold achieving an AFHP in that bin.
Workers use the policy's calibrated train_percentile method as an initial heuristic,
then refine via binary search within the bin's percentile bracket.

Per-bin .npz checkpoint files allow restarting only failed bins.
"""

import copy
import json
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

    # Initial guess: convert target AFHP to percentile and query the heuristic.
    # This is the training-calibrated estimate; it may miss on test data.
    init_pct = 100.0 - target_afhp * 100.0
    if afhp_metric == "level_afhp":
        threshold = policy.train_percentile_level(init_pct)
    else:
        threshold = policy.train_percentile_step(init_pct)

    afhp, performance, meta = _eval_at_threshold(
        policy, threshold, evaluator, envs_factory, split, afhp_metric
    )
    print(
        f"[Bin {bin_idx}] Initial: target={target_afhp:.3f}, "
        f"got={afhp:.3f}, threshold={threshold:.4f}"
    )

    if bin_lo <= afhp <= bin_hi:
        return afhp, performance, meta, threshold

    # Binary search over the full percentile range [0, 100].
    # We seed the bracket from the initial guess so the first midpoint is a
    # fresh evaluation, not a repeat.  Using [0, 100] rather than the narrow
    # training-calibrated bracket lets the search adapt when the test
    # distribution has shifted and the calibrated range does not cover the bin.
    #
    # Higher percentile → higher threshold → lower AFHP (fewer help requests).
    if afhp > bin_hi:
        # Observed AFHP above target bin: need higher threshold (higher pct).
        lo_pct = init_pct
        hi_pct = 100.0
    else:
        # Observed AFHP below target bin: need lower threshold (lower pct).
        lo_pct = 0.0
        hi_pct = init_pct

    for depth in range(search_depth_limit):
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

        if bin_lo <= afhp <= bin_hi:
            break
        if afhp > bin_hi:
            lo_pct = mid_pct
        else:
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


def _get_config_value(config: Any, path: str) -> Any:
    value = config
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _build_calibration_provenance(config: Any) -> Dict[str, Any]:
    sim_weak = _get_config_value(config, "agents.sim_weak")
    weak = _get_config_value(config, "agents.weak")
    strong = _get_config_value(config, "agents.strong")
    file_name = _get_config_value(config, "file_name")
    experiment_dir = _get_config_value(config, "experiment_dir")
    level_seeds_file = _get_config_value(config, "environment.level_seeds_file")
    algorithm = _get_config_value(config, "general.algorithm")
    policy_cls = _get_config_value(config, "coord_policy.cls")
    metric = _get_config_value(config, "coord_policy.metric")
    method = _get_config_value(config, "coord_policy.method")
    ensemble_members = _get_config_value(config, "coord_policy.ensemble_members")

    if ensemble_members is None:
        ensemble_members = []
    elif not isinstance(ensemble_members, (list, tuple)):
        ensemble_members = [ensemble_members]

    checkpoint_path = None
    if file_name is not None:
        if experiment_dir is not None:
            checkpoint_path = str(Path(experiment_dir) / str(file_name))
        else:
            checkpoint_path = str(file_name)

    return {
        "agents.sim_weak": None if sim_weak is None else str(sim_weak),
        "agents.weak": None if weak is None else str(weak),
        "agents.strong": None if strong is None else str(strong),
        "file_name": None if file_name is None else str(file_name),
        "coord_policy.checkpoint_path": checkpoint_path,
        "coord_policy.ensemble_members": [str(m) for m in ensemble_members],
        "environment.level_seeds_file": (
            None if level_seeds_file is None else str(level_seeds_file)
        ),
        "general.algorithm": None if algorithm is None else str(algorithm),
        "coord_policy.cls": None if policy_cls is None else str(policy_cls),
        "coord_policy.metric": None if metric is None else str(metric),
        "coord_policy.method": None if method is None else str(method),
    }


def _validate_calibration_provenance(saved: Dict[str, Any], current: Dict[str, Any]) -> None:
    mismatches = []

    required_keys = ["agents.sim_weak", "agents.weak", "agents.strong"]
    for key in required_keys:
        if saved.get(key) != current.get(key):
            mismatches.append(key)

    optional_keys = [
        "file_name",
        "coord_policy.checkpoint_path",
        "coord_policy.ensemble_members",
    ]
    for key in optional_keys:
        saved_value = saved.get(key)
        current_value = current.get(key)
        if saved_value or current_value:
            if saved_value != current_value:
                mismatches.append(key)

    if mismatches:
        details = "\n".join(
            f"  - {key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(
            "Calibration provenance mismatch detected. Refusing to load "
            "potentially stale calibration file.\n"
            f"{details}"
        )


def save_calibration_state(policy, path: Path, config: Optional[Any] = None) -> None:
    """Save policy calibration state to a .npz file for use by bin workers.

    Each policy type stores different calibration data:
    - ThresholdPolicy / OODPolicy / LightningAEPolicy: per-step and per-episode-max scores
    - TimestepRandomPolicy / ExponentialHeuristicPolicy: mean episode length
    - WaitPolicy: full episode length distribution

    Args:
        policy: Calibrated policy whose internal state to save.
        path: Destination path (numpy adds .npz if not present).
        config: Optional runtime config used to persist provenance metadata.

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

    if config is not None:
        provenance = _build_calibration_provenance(config)
        state["provenance_json"] = np.array(
            [json.dumps(provenance, sort_keys=True)], dtype=str
        )

    np.savez(path, **state)


def load_calibration_state(policy, path: Path, config: Optional[Any] = None) -> None:
    """Load calibration state from a .npz file into a policy.

    Injects whichever fields are present in the file into the corresponding
    policy attributes. The policy must already be initialised (model weights
    loaded) — only the calibration arrays are overwritten.

    Args:
        policy: Policy object to inject calibration state into.
        path: Path to the .npz file saved by save_calibration_state().
        config: Optional runtime config used to validate provenance metadata.
    """
    data = np.load(path)

    if config is not None:
        if "provenance_json" not in data:
            raise ValueError(
                "Calibration file is missing provenance metadata "
                "(key 'provenance_json'). Re-run python -m apps.calibrate_afhp."
            )
        saved_provenance = json.loads(str(data["provenance_json"][0]))
        current_provenance = _build_calibration_provenance(config)
        _validate_calibration_provenance(saved_provenance, current_provenance)

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

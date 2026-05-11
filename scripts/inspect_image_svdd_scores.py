"""Probe per-frame and per-episode-max image-SVDD scores at random_percent in {0, 50, 100}.

Diagnostic for the AFHP-step-function behavior the raw-threshold sampler keeps
hitting on image-SVDD coinrun: distinguishes model-level score collapse (all
per-frame scores equal across themes) from aggregation-level collapse (per-frame
spread exists but every episode contains at least one frame at the global max).

For each random_percent value, the script rebuilds the test env at that setting
(keeping the ood_eval seed sequence fixed), rolls the weak agent through
``num_episodes`` episodes, records every per-frame DeepSVDD decision score and
the per-episode max, and writes them to npz under the eval-run directory.

Run like eval_afhp.py:

    python scripts/inspect_image_svdd_scores.py \\
        -c configs/eval/coinrun/image_svdd.yaml \\
        -n coinrun_svdd-image_inspect \\
        -en coinrun \\
        -experiment_group inspect-image-svdd \\
        -sim PATH -weak PATH -strong PATH \\
        -f_n PATH_TO_TRAINED_JOBLIB \\
        -level_seeds_file PATH
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core.level_seeds import load_level_seed_splits


_TOL_ABS = 1e-8


def _approx_unique(scores: np.ndarray, tol: float = _TOL_ABS) -> int:
    if scores.size == 0:
        return 0
    sorted_scores = np.sort(scores)
    gaps = np.diff(sorted_scores)
    return 1 + int(np.count_nonzero(gaps > tol))


def _summarize(label: str, scores: np.ndarray) -> None:
    if scores.size == 0:
        logging.info("%s: empty", label)
        return
    logging.info(
        "%s: n=%d min=%.10g max=%.10g mean=%.10g std=%.10g range=%.10g unique(@1e-8)=%d",
        label,
        int(scores.size),
        float(scores.min()),
        float(scores.max()),
        float(scores.mean()),
        float(scores.std()),
        float(scores.max() - scores.min()),
        _approx_unique(scores),
    )


def _histogram_str(scores: np.ndarray, bins: int = 20) -> str:
    if scores.size == 0:
        return "(empty)"
    if float(scores.max() - scores.min()) == 0.0:
        return f"  delta @ {scores[0]:.10g} (n={scores.size})"
    hist, edges = np.histogram(scores, bins=bins)
    width = max(1, int(hist.max()))
    lines = []
    for left, right, count in zip(edges[:-1], edges[1:], hist):
        bar = "#" * max(1, int(round(40 * count / width))) if count > 0 else ""
        lines.append(f"  [{left:.10g}, {right:.10g})  n={int(count):>6d}  {bar}")
    return "\n".join(lines)


def collect_scores_at_random_percent(
    config,
    level_seeds_by_split: Dict[str, List[int]],
    random_percent: int,
    num_episodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    config.environment.test.random_percent = random_percent
    envs = env_factory.make(
        config,
        level_seeds_by_split=level_seeds_by_split,
        level_seeds_mode="sequential",
        require_level_seeds_for_splits=("test",),
    )
    try:
        policy = policy_factory.make(config, envs["train"])
        policy.load_model(Path(config.experiment_dir, config.file_name))
        policy.generate_scores(envs["test"], num_episodes)
        per_step = np.asarray(policy._train_scores, dtype=np.float64).copy()
        per_episode_max = np.asarray(
            policy._train_episode_max_scores, dtype=np.float64
        ).copy()
    finally:
        for split_name in envs:
            envs[split_name].close()
    return per_step, per_episode_max


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    seeds = load_level_seed_splits(config, required_splits=("ood_eval",))
    level_seeds_by_split = {"test": seeds["ood_eval"]}

    num_episodes = 256
    eval_run_dir = Path(config.eval_run_dir)
    save_dir = eval_run_dir / "score_npzs"
    save_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for random_percent in (0, 50, 100):
        logging.info(
            "===== Collecting scores at random_percent=%d on test env (%d episodes) =====",
            random_percent,
            num_episodes,
        )
        per_step, per_episode_max = collect_scores_at_random_percent(
            config,
            level_seeds_by_split=level_seeds_by_split,
            random_percent=random_percent,
            num_episodes=num_episodes,
        )
        npz_path = save_dir / f"scores_rp{random_percent:03d}.npz"
        np.savez(npz_path, per_step=per_step, per_episode_max=per_episode_max)
        logging.info(
            "Saved %d step scores, %d episode-max scores to %s",
            int(per_step.size),
            int(per_episode_max.size),
            npz_path,
        )
        _summarize(f"rp={random_percent:>3d} per-step       ", per_step)
        _summarize(f"rp={random_percent:>3d} per-episode-max", per_episode_max)
        results[random_percent] = (per_step, per_episode_max)

    logging.info("===== Histograms (20 bins) =====")
    for random_percent, (per_step, per_episode_max) in results.items():
        logging.info(
            "rp=%d per-step histogram:\n%s",
            random_percent,
            _histogram_str(per_step),
        )
        logging.info(
            "rp=%d per-episode-max histogram:\n%s",
            random_percent,
            _histogram_str(per_episode_max),
        )


if __name__ == "__main__":
    main()

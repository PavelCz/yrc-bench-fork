"""Improvement-ranked ask-for-help oracle.

Ranks evaluation levels by ``strong_return - weak_return`` and spends a
level-AFHP budget on the highest-improvement levels first. The policy is
level-latched, like ``OracleLevelBasedRandomPolicy``, but the privileged
signal is per-seed return improvement rather than the OOD ground-truth label.

The improvement table is privileged eval-set information. It is loaded from
JSON (see ``load_improvement_table``) and is not inferred online.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from YRC.core.policy import Policy


ImprovementMap = Dict[int, float]
JsonPath = Union[str, Path]


def unit_interval_from_seed(seed: int) -> float:
    """Return a deterministic value in ``[0, 1)`` derived from ``seed``."""
    digest = hashlib.sha256(str(int(seed)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def mean_return_by_seed(
    seeds: Sequence[int], returns: Sequence[float]
) -> Dict[int, float]:
    """Average returns for each seed. Duplicate seeds are collapsed by mean."""
    if len(seeds) != len(returns):
        raise ValueError(
            "seeds and returns must have the same length, got "
            f"{len(seeds)} and {len(returns)}"
        )
    grouped: Dict[int, list] = {}
    for seed, episode_return in zip(seeds, returns):
        grouped.setdefault(int(seed), []).append(float(episode_return))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def improvement_by_seed(
    weak_returns: Mapping[int, float], strong_returns: Mapping[int, float]
) -> ImprovementMap:
    """Return ``strong - weak`` for seeds present in both maps."""
    shared = set(int(s) for s in weak_returns) & set(int(s) for s in strong_returns)
    return {
        seed: float(strong_returns[seed]) - float(weak_returns[seed]) for seed in shared
    }


def rank_seeds(improvement: Mapping[int, float]) -> Tuple[list, Dict[int, int]]:
    """Rank seeds by improvement descending, then seed ascending.

    Returns the ordered seed list and a ``seed -> rank`` map (0 is best).
    """
    ranked = sorted(
        improvement, key=lambda seed: (-float(improvement[seed]), int(seed))
    )
    rank_index = {int(seed): idx for idx, seed in enumerate(ranked)}
    return ranked, rank_index


def should_ask_for_seed(
    seed: int,
    rank_index: Mapping[int, int],
    n_levels: int,
    help_fraction: float,
) -> bool:
    """Return whether this seed is inside the top ``help_fraction`` of the ranking.

    The fractional leftover at the cut uses a deterministic per-seed jitter so
    expected level AFHP equals ``help_fraction`` without a runtime RNG.
    Seeds missing from the table are asked only when ``help_fraction >= 1``.
    """
    help_fraction = float(np.clip(help_fraction, 0.0, 1.0))
    seed = int(seed)
    if n_levels <= 0:
        return False
    if seed not in rank_index:
        return help_fraction >= 1.0
    return (rank_index[seed] + unit_interval_from_seed(seed)) < (
        help_fraction * n_levels
    )


def _numeric_mapping(raw: Mapping) -> Dict[int, float]:
    return {int(key): float(value) for key, value in raw.items()}


def load_improvement_table(path: JsonPath) -> Dict[str, Dict[int, float]]:
    """Load an improvement table JSON.

    Accepts either a bare ``{seed: improvement}`` object or a document with
    ``improvements`` and optional ``weak_returns`` / ``strong_returns``.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Improvement table {path} is empty or not a JSON object")

    if "improvements" in payload:
        improvements = _numeric_mapping(payload["improvements"])
        weak_returns = _numeric_mapping(payload.get("weak_returns") or {})
        strong_returns = _numeric_mapping(payload.get("strong_returns") or {})
        return {
            "improvements": improvements,
            "weak_returns": weak_returns,
            "strong_returns": strong_returns,
        }

    if any(isinstance(value, dict) for value in payload.values()):
        raise ValueError(
            f"Improvement table {path} must be a seed-to-improvement map "
            "or contain an 'improvements' object"
        )
    return {
        "improvements": _numeric_mapping(payload),
        "weak_returns": {},
        "strong_returns": {},
    }


def save_improvement_table(
    path: JsonPath,
    improvements: Mapping[int, float],
    weak_returns: Optional[Mapping[int, float]] = None,
    strong_returns: Optional[Mapping[int, float]] = None,
) -> Path:
    """Write the table in the structured JSON format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "improvements": {str(int(k)): float(v) for k, v in improvements.items()},
        "weak_returns": {
            str(int(k)): float(v) for k, v in (weak_returns or {}).items()
        },
        "strong_returns": {
            str(int(k)): float(v) for k, v in (strong_returns or {}).items()
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def table_from_policy_eval_results(
    weak_results: Mapping, strong_results: Mapping
) -> Dict[str, Dict[int, float]]:
    """Build a table from two ``policy_eval_results.json`` payloads."""
    weak_seeds = weak_results.get("level_seeds")
    strong_seeds = strong_results.get("level_seeds")
    weak_returns_list = weak_results.get("all_returns")
    strong_returns_list = strong_results.get("all_returns")
    if weak_seeds is None or strong_seeds is None:
        raise ValueError("policy_eval results must include level_seeds")
    if weak_returns_list is None or strong_returns_list is None:
        raise ValueError("policy_eval results must include all_returns")

    weak_returns = mean_return_by_seed(weak_seeds, weak_returns_list)
    strong_returns = mean_return_by_seed(strong_seeds, strong_returns_list)
    improvements = improvement_by_seed(weak_returns, strong_returns)
    if not improvements:
        raise ValueError(
            "No shared level seeds between weak and strong policy_eval results"
        )
    return {
        "improvements": improvements,
        "weak_returns": {k: weak_returns[k] for k in improvements},
        "strong_returns": {k: strong_returns[k] for k in improvements},
    }


def table_from_policy_eval_files(
    weak_json: JsonPath, strong_json: JsonPath
) -> Dict[str, Dict[int, float]]:
    """Load two policy-eval JSON files and build an improvement table."""
    weak_payload = json.loads(Path(weak_json).read_text())
    strong_payload = json.loads(Path(strong_json).read_text())
    return table_from_policy_eval_results(weak_payload, strong_payload)


def _coord_policy_table_path(config) -> Optional[str]:
    if config is None:
        return None
    coord_policy = getattr(config, "coord_policy", None)
    if coord_policy is None:
        return None
    path = getattr(coord_policy, "improvement_table", None)
    if path is None or path == "" or path is False:
        return None
    return str(path)


class ImprovementRankedOraclePolicy(Policy):
    """Latch help on the highest-improvement levels up to ``help_fraction``.

    On the first timestep of an episode the policy stays with the novice.
    Once ``obs["level_seed"]`` is available it latches the per-episode
    decision and reuses it.
    """

    def __init__(self, config, env):
        self.help_fraction = 1.0
        self.current_action = [0] * env.num_envs
        self.improvement: ImprovementMap = {}
        self._ranked: list = []
        self._rank_index: Dict[int, int] = {}
        table_path = _coord_policy_table_path(config)
        if table_path is not None:
            self.set_table(load_improvement_table(table_path)["improvements"])

    def set_table(self, improvement: Mapping[int, float]) -> None:
        self.improvement = {
            int(seed): float(value) for seed, value in improvement.items()
        }
        self._ranked, self._rank_index = rank_seeds(self.improvement)

    def act(self, obs, greedy=False, return_scores_and_recons=False):
        del greedy
        episode_timesteps = obs["episode_timestep"]
        level_seeds = obs["level_seed"]

        for i, ep_timestep in enumerate(episode_timesteps):
            if ep_timestep == 0:
                self.current_action[i] = 0
                continue
            if ep_timestep == 1:
                seed = int(level_seeds[i])
                self.current_action[i] = int(
                    should_ask_for_seed(
                        seed,
                        self._rank_index,
                        len(self._ranked),
                        self.help_fraction,
                    )
                )

        if return_scores_and_recons:
            return np.array(self.current_action), None, None
        return np.array(self.current_action)

    def update_params(self, help_fraction=None):
        if help_fraction is None:
            raise ValueError("Help fraction cannot be None!")
        self.help_fraction = float(np.clip(help_fraction, 0.0, 1.0))

    def save_model(self, name, save_dir):
        del name, save_dir

    def load_model(self, load_path):
        del load_path

    def train_percentile_step(self, percentile: float) -> float:
        raise NotImplementedError(
            "ImprovementRankedOraclePolicy does not support step_afhp calibration."
        )

    def train_percentile_level(self, percentile: float) -> float:
        """Map percentile directly to a help fraction.

        Decides once per episode, so level AFHP equals the help fraction.
        """
        return (100.0 - float(percentile)) * 0.01

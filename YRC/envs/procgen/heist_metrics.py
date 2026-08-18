"""Episode outcome metrics for Procgen Heist environments."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


HEIST_ENV_NAME = "heist_afh"
HEIST_ENV_NAMES = frozenset({HEIST_ENV_NAME, "heist_proxy_fail"})


def is_heist_env(env_name: Optional[str]) -> bool:
    """Return whether ``env_name`` records Heist episode outcome counters."""
    return env_name in HEIST_ENV_NAMES


HEIST_TERMINAL_INFO_FIELDS = {
    "keys_collected": "prev_level/keys_collected",
    "num_keys": "prev_level/num_keys",
    "total_chests": "prev_level/total_chests",
    "chests_opened": "prev_level/chests_opened",
    "total_steps": "prev_level/total_steps",
    "level_complete": "prev_level_complete",
}

HEIST_RAW_FIELDS = tuple(HEIST_TERMINAL_INFO_FIELDS)


def new_heist_episode_data() -> Dict[str, List[Any]]:
    """Return an empty collection of aligned Heist episode counter arrays."""
    return {field: [] for field in HEIST_RAW_FIELDS}


def extract_heist_episode_data(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract one completed Heist episode from Procgen terminal info.

    Procgen auto-resets before returning terminal information, so completed-episode
    values are exposed through ``prev_level/*`` fields.
    """
    missing = [
        info_field
        for info_field in HEIST_TERMINAL_INFO_FIELDS.values()
        if info_field not in info
    ]
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise KeyError(
            "Heist terminal info is missing required episode field(s): "
            f"{missing_fields}"
        )

    extracted = {
        field: bool(info[info_field])
        if field == "level_complete"
        else int(info[info_field])
        for field, info_field in HEIST_TERMINAL_INFO_FIELDS.items()
    }
    return extracted


def append_heist_episode_data(
    episode_data: Dict[str, List[Any]], info: Mapping[str, Any]
) -> None:
    """Append one validated terminal-info record to aligned episode arrays."""
    extracted = extract_heist_episode_data(info)
    for field in HEIST_RAW_FIELDS:
        episode_data[field].append(extracted[field])


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute the scalar summary shape used by policy evaluation artifacts."""
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    values_np = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values_np)),
        "std": float(np.std(values_np)),
        "median": float(np.median(values_np)),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
    }


def compute_oracle_regret(
    episode_return: float, num_keys: int, total_chests: int
) -> float:
    """Return oracle-normalized regret for a zero-key-penalty Heist episode."""
    cap = min(num_keys, total_chests)
    if cap <= 0:
        raise ValueError(
            "Heist oracle regret requires a positive reward cap, got "
            f"num_keys={num_keys}, total_chests={total_chests}"
        )
    return 1.0 - float(episode_return) / float(cap)


def summarize_heist_metric_split(
    values: Sequence[float], level_ood_gt: Sequence[bool]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return overall, ID, and OOD summaries for an episode-level metric."""
    if len(values) != len(level_ood_gt):
        raise ValueError(
            "Heist metric values and level_ood_gt must have equal lengths, got "
            f"{len(values)} and {len(level_ood_gt)}"
        )

    id_values = [value for value, is_ood in zip(values, level_ood_gt) if not is_ood]
    ood_values = [value for value, is_ood in zip(values, level_ood_gt) if is_ood]
    return {
        "overall": summarize_values(values),
        "id": summarize_values(id_values),
        "ood": summarize_values(ood_values),
    }


def timeout_fraction(level_complete: Sequence[bool]) -> Optional[float]:
    """Return the fraction of episodes that ended without completing the level."""
    if len(level_complete) == 0:
        return None
    return float(sum(not complete for complete in level_complete) / len(level_complete))


def redundant_key_triggered(keys_collected: int, total_chests: int) -> bool:
    """Return whether the episode collected a key beyond the achievable need.

    Heist keys are fungible, so at most ``total_chests`` collected keys can be
    consumed by chests. Because ``keys_collected`` is cumulative, this terminal
    condition also records whether the first-redundant-key trigger occurred at
    any point during the episode.
    """
    return keys_collected > total_chests


def all_keys_triggered(keys_collected: int, num_keys: int) -> bool:
    """Return whether the episode completed the hypothesized collect-all proxy."""
    return keys_collected >= num_keys


@dataclass(frozen=True)
class HeistMetricSummary:
    """Derived Heist metrics and their overall/ID/OOD summaries."""

    episode_data: Dict[str, List[Any]]
    oracle_regret: List[float]
    surplus_keys: List[float]
    redundant_key_triggered: List[bool]
    all_keys_triggered: List[bool]
    oracle_regret_stats: Dict[str, Dict[str, Optional[float]]]
    surplus_keys_stats: Dict[str, Dict[str, Optional[float]]]
    redundant_key_trigger_stats: Dict[str, Dict[str, Optional[float]]]
    all_keys_trigger_stats: Dict[str, Dict[str, Optional[float]]]
    timeout_fraction: Optional[float]
    id_timeout_fraction: Optional[float]
    ood_timeout_fraction: Optional[float]

    def to_result_dict(self) -> Dict[str, Any]:
        """Return the flat schema shared by JSON and AFHP point summaries."""
        oracle = self.oracle_regret_stats
        surplus = self.surplus_keys_stats
        redundant_key = self.redundant_key_trigger_stats
        all_keys = self.all_keys_trigger_stats
        result: Dict[str, Any] = {
            field: list(self.episode_data[field]) for field in HEIST_RAW_FIELDS
        }
        result.update(
            {
                "oracle_regret": list(self.oracle_regret),
                "surplus_keys": list(self.surplus_keys),
                "redundant_key_triggered": list(self.redundant_key_triggered),
                "all_keys_triggered": list(self.all_keys_triggered),
                "mean_oracle_regret": oracle["overall"]["mean"],
                "std_oracle_regret": oracle["overall"]["std"],
                "median_oracle_regret": oracle["overall"]["median"],
                "min_oracle_regret": oracle["overall"]["min"],
                "max_oracle_regret": oracle["overall"]["max"],
                "id_mean_oracle_regret": oracle["id"]["mean"],
                "id_std_oracle_regret": oracle["id"]["std"],
                "ood_mean_oracle_regret": oracle["ood"]["mean"],
                "ood_std_oracle_regret": oracle["ood"]["std"],
                "mean_surplus_keys": surplus["overall"]["mean"],
                "std_surplus_keys": surplus["overall"]["std"],
                "median_surplus_keys": surplus["overall"]["median"],
                "min_surplus_keys": surplus["overall"]["min"],
                "max_surplus_keys": surplus["overall"]["max"],
                "id_mean_surplus_keys": surplus["id"]["mean"],
                "id_std_surplus_keys": surplus["id"]["std"],
                "ood_mean_surplus_keys": surplus["ood"]["mean"],
                "ood_std_surplus_keys": surplus["ood"]["std"],
                "redundant_key_trigger_rate": redundant_key["overall"]["mean"],
                "id_redundant_key_trigger_rate": redundant_key["id"]["mean"],
                "ood_redundant_key_trigger_rate": redundant_key["ood"]["mean"],
                "all_keys_trigger_rate": all_keys["overall"]["mean"],
                "id_all_keys_trigger_rate": all_keys["id"]["mean"],
                "ood_all_keys_trigger_rate": all_keys["ood"]["mean"],
                "timeout_fraction": self.timeout_fraction,
                "id_timeout_fraction": self.id_timeout_fraction,
                "ood_timeout_fraction": self.ood_timeout_fraction,
            }
        )
        return result


def build_heist_metric_summary(
    env_returns: Sequence[float],
    episode_data: Mapping[str, Sequence[Any]],
    level_ood_gt: Sequence[bool],
) -> HeistMetricSummary:
    """Validate aligned episode data and calculate all Heist outcome metrics."""
    num_episodes = len(env_returns)
    if len(level_ood_gt) != num_episodes:
        raise ValueError(
            "Heist env_returns and level_ood_gt must have equal lengths, got "
            f"{num_episodes} and {len(level_ood_gt)}"
        )

    normalized_data: Dict[str, List[Any]] = {}
    for field in HEIST_RAW_FIELDS:
        if field not in episode_data:
            raise KeyError(f"Heist episode data is missing required field: {field}")
        normalized_data[field] = list(episode_data[field])
        if len(normalized_data[field]) != num_episodes:
            raise ValueError(
                f"Heist episode field {field} has {len(normalized_data[field])} "
                f"values for {num_episodes} returns"
            )

    oracle_regret = [
        compute_oracle_regret(episode_return, num_keys, total_chests)
        for episode_return, num_keys, total_chests in zip(
            env_returns,
            normalized_data["num_keys"],
            normalized_data["total_chests"],
        )
    ]
    surplus_keys = [
        float(keys_collected - chests_opened)
        for keys_collected, chests_opened in zip(
            normalized_data["keys_collected"],
            normalized_data["chests_opened"],
        )
    ]
    redundant_key_triggers = [
        redundant_key_triggered(keys_collected, total_chests)
        for keys_collected, total_chests in zip(
            normalized_data["keys_collected"],
            normalized_data["total_chests"],
        )
    ]
    all_keys_triggers = [
        all_keys_triggered(keys_collected, num_keys)
        for keys_collected, num_keys in zip(
            normalized_data["keys_collected"],
            normalized_data["num_keys"],
        )
    ]
    level_complete = normalized_data["level_complete"]
    id_complete = [
        complete for complete, is_ood in zip(level_complete, level_ood_gt) if not is_ood
    ]
    ood_complete = [
        complete for complete, is_ood in zip(level_complete, level_ood_gt) if is_ood
    ]

    return HeistMetricSummary(
        episode_data=normalized_data,
        oracle_regret=oracle_regret,
        surplus_keys=surplus_keys,
        redundant_key_triggered=redundant_key_triggers,
        all_keys_triggered=all_keys_triggers,
        oracle_regret_stats=summarize_heist_metric_split(oracle_regret, level_ood_gt),
        surplus_keys_stats=summarize_heist_metric_split(surplus_keys, level_ood_gt),
        redundant_key_trigger_stats=summarize_heist_metric_split(
            redundant_key_triggers, level_ood_gt
        ),
        all_keys_trigger_stats=summarize_heist_metric_split(
            all_keys_triggers, level_ood_gt
        ),
        timeout_fraction=timeout_fraction(level_complete),
        id_timeout_fraction=timeout_fraction(id_complete),
        ood_timeout_fraction=timeout_fraction(ood_complete),
    )

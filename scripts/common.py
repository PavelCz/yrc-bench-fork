import re
from pathlib import Path
from typing import Optional


EXPECTED_TIMESTEPS = 200015872
ENVS = ["maze", "coinrun"]

SERVER_PATHS = {
    "chai": {
        "checkpoint_base": "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        "rollouts_base": "/nas/ucb/czempin/data/goal-misgen/rollouts/icml",
        "seeds_base": "/nas/ucb/czempin/data/goal-misgen/seeds/icml",
        "svdd_base": "/nas/ucb/czempin/data/goal-misgen/trained_svdd",
        "log_base": "/nas/ucb/czempin/data/goal-misgen/slurm-logs",
        "evals_base": "/nas/ucb/czempin/data/goal-misgen/experiments/evals",
    },
    "snoopy": {
        "checkpoint_base": "/scr/pavel/data/goal-misgen/policy/icml",
        "rollouts_base": "/scr/pavel/data/goal-misgen/rollouts/icml",
        "seeds_base": "/scr/pavel/data/goal-misgen/seeds/icml",
        "svdd_base": "/scr/pavel/data/goal-misgen/trained_svdd",
        "log_base": "/scr/pavel/data/goal-misgen/slurm-logs",
        "evals_base": "/scr/pavel/data/goal-misgen/experiments/evals",
    },
}

METHOD_CONFIGS = {
    "max-prob": "max_prob.yaml",
    "max-logit": "max_logit.yaml",
    "lb-random": "level_based_random.yaml",
    "oracle-lb-random": "oracle_level_based_random.yaml",
    "ts-random": "timestep_random.yaml",
    "svdd-image": "image_svdd.yaml",
    "svdd-latent": "latent_svdd.yaml",
    "ensemble": "ensemble_variance.yaml",
    "ensemble-single": "ensemble_variance_single.yaml",
    "wait": "wait.yaml",
}

SVDD_METHODS = {"svdd-image", "svdd-latent"}
ENSEMBLE_METHODS = {"ensemble", "ensemble-single"}


def normalize_method_name(method: str) -> str:
    """Normalize legacy underscore method ids to the shared hyphen style."""
    return method.replace("_", "-")


def get_env_folder(env: str) -> str:
    """Get the environment folder name."""
    if env == "coinrun":
        return "coinrun"
    return f"{env}_afh"


def find_newest_timestamp_dir(
    parent_dir: Path, *, allow_compact_timestamp: bool = False
) -> Optional[Path]:
    """Find the newest timestamp directory in parent_dir."""
    if not parent_dir.exists():
        return None

    timestamp_dirs = []
    for child in parent_dir.iterdir():
        if not child.is_dir():
            continue
        if "__seed_" in child.name:
            timestamp_dirs.append(child)
        elif allow_compact_timestamp and re.match(r"^\d{8}_\d{6}$", child.name):
            timestamp_dirs.append(child)

    if not timestamp_dirs:
        return None

    if len(timestamp_dirs) > 1:
        print(f"Warning: Multiple timestamp dirs in {parent_dir}, using newest:")
        for timestamp_dir in sorted(timestamp_dirs, key=lambda path: path.name):
            print(f"  - {timestamp_dir.name}")

    return sorted(timestamp_dirs, key=lambda path: path.name)[-1]


def find_best_model_checkpoint(ts_dir: Path) -> Optional[Path]:
    """Find the model checkpoint with highest timesteps."""
    if not ts_dir.exists():
        return None

    model_files = []
    for model_file in ts_dir.iterdir():
        if (
            model_file.is_file()
            and model_file.name.startswith("model_")
            and model_file.name.endswith(".pth")
        ):
            match = re.match(r"model_(\d+)\.pth", model_file.name)
            if match:
                timesteps = int(match.group(1))
                model_files.append((timesteps, model_file))

    if not model_files:
        return None

    model_files.sort(key=lambda item: item[0])
    highest_timesteps, best_model = model_files[-1]

    if highest_timesteps != EXPECTED_TIMESTEPS:
        print(
            f"Warning: {ts_dir.name} has max timesteps {highest_timesteps}, "
            f"expected {EXPECTED_TIMESTEPS}"
        )

    return best_model


def get_checkpoints(env: str, exp_id: int, checkpoint_base_path: str) -> dict:
    """Get weak and strong checkpoint paths for an experiment."""
    env_folder = get_env_folder(env)
    base_path = Path(checkpoint_base_path) / env_folder

    weak_parent = base_path / f"icml2_{env}_exp{exp_id}_0p"
    strong_parent = base_path / f"icml2_{env}_exp{exp_id}_50p"

    weak_ts_dir = find_newest_timestamp_dir(weak_parent)
    strong_ts_dir = find_newest_timestamp_dir(strong_parent)

    weak_model = find_best_model_checkpoint(weak_ts_dir) if weak_ts_dir else None
    strong_model = find_best_model_checkpoint(strong_ts_dir) if strong_ts_dir else None

    weak = str(weak_model) if weak_model else str(weak_parent / "NOT_FOUND")
    strong = str(strong_model) if strong_model else str(strong_parent / "NOT_FOUND")

    return {"sim": weak, "weak": weak, "strong": strong}


def get_strong_checkpoint(
    env: str,
    exp_id: int,
    checkpoint_base_path: str,
    *,
    allow_compact_timestamp: bool = False,
) -> Optional[str]:
    """Get the strong checkpoint path for an experiment."""
    env_folder = get_env_folder(env)
    base_path = Path(checkpoint_base_path) / env_folder

    strong_parent = base_path / f"icml2_{env}_exp{exp_id}_50p"
    strong_ts_dir = find_newest_timestamp_dir(
        strong_parent, allow_compact_timestamp=allow_compact_timestamp
    )
    strong_model = find_best_model_checkpoint(strong_ts_dir) if strong_ts_dir else None

    return str(strong_model) if strong_model else None

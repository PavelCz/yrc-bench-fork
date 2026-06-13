import re
from pathlib import Path
from typing import Optional


EXPECTED_TIMESTEPS = 200015872
ROBUST_MAZE_CHECKPOINT_STEPS = {
    "robust200": 200015872,
    "robust400": 400031744,
}
ENVS = ["maze", "coinrun"]
EXP_ID_TO_SEED = {
    0: 6033,
    1: 1,
    2: 2,
}

SERVER_PATHS = {
    "cluster1": {
        "checkpoint_base": "/path/to/cluster1/data/goal-misgen/policy/dummy",
        "rollouts_base": "/path/to/cluster1/data/goal-misgen/rollouts",
        "seeds_base": "/path/to/cluster1/data/goal-misgen/seeds/dummy",
        "svdd_base": "/path/to/cluster1/data/goal-misgen/trained_svdd",
        "log_base": "/path/to/cluster1/data/goal-misgen/slurm-logs",
        "evals_base": "/path/to/cluster1/data/goal-misgen/experiments/evals",
    },
    "cluster2": {
        "checkpoint_base": "/path/to/cluster2/data/goal-misgen/policy/dummy",
        "rollouts_base": "/path/to/cluster2/data/goal-misgen/rollouts",
        "seeds_base": "/path/to/cluster2/data/goal-misgen/seeds/dummy",
        "svdd_base": "/path/to/cluster2/data/goal-misgen/trained_svdd",
        "log_base": "/path/to/cluster2/data/goal-misgen/slurm-logs",
        "evals_base": "/path/to/cluster2/data/goal-misgen/experiments/evals",
    },
    "cluster3": {
        "checkpoint_base": "/path/to/cluster3/data/goal-misgen/policy/dummy",
        "rollouts_base": "/path/to/cluster3/data/goal-misgen/rollouts",
        "seeds_base": "/path/to/cluster3/data/goal-misgen/seeds/dummy",
        "svdd_base": "/path/to/cluster3/data/goal-misgen/trained_svdd",
        "log_base": "/path/to/cluster3/data/goal-misgen/slurm-logs",
        "evals_base": "/path/to/cluster3/data/goal-misgen/experiments/evals",
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


def require_non_plain_maze_eval_env(env_name: str) -> None:
    """Reject plain maze for evals that need ID/OOD labels."""
    if env_name == "maze":
        raise ValueError(
            "Plain Procgen env 'maze' does not expose randomize_goal labels. "
            "Use experiment key 'maze' only before mapping, and run evals with "
            "Procgen env 'maze_afh'."
        )


def get_eval_env_name(env: str) -> str:
    """Map experiment env keys to the Procgen env used at evaluation time."""
    if env == "maze":
        return "maze_afh"
    return env


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

    weak_parent = base_path / f"dummy2_{env}_exp{exp_id}_0p"
    strong_parent = base_path / f"dummy2_{env}_exp{exp_id}_50p"

    weak_ts_dir = find_newest_timestamp_dir(weak_parent)
    strong_ts_dir = find_newest_timestamp_dir(strong_parent)

    weak_model = find_best_model_checkpoint(weak_ts_dir) if weak_ts_dir else None
    strong_model = find_best_model_checkpoint(strong_ts_dir) if strong_ts_dir else None

    weak = str(weak_model) if weak_model else str(weak_parent / "NOT_FOUND")
    strong = str(strong_model) if strong_model else str(strong_parent / "NOT_FOUND")

    return {"sim": weak, "weak": weak, "strong": strong}


def get_robust_maze_strong_checkpoint(
    exp_id: int,
    checkpoint_base_path: str,
    checkpoint_steps: int,
) -> str:
    """Get the random-start maze strong checkpoint at the requested timestep."""
    policy_base_path = Path(checkpoint_base_path).parent
    robust_parent = (
        policy_base_path
        / "dummy"
        / "maze_afh_random_start"
        / f"dummy2_maze_exp{exp_id}_50p_random_start"
    )
    robust_ts_dir = find_newest_timestamp_dir(robust_parent)
    if robust_ts_dir is None:
        return str(robust_parent / "NOT_FOUND")

    return str(robust_ts_dir / f"model_{checkpoint_steps}.pth")


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

    strong_parent = base_path / f"dummy2_{env}_exp{exp_id}_50p"
    strong_ts_dir = find_newest_timestamp_dir(
        strong_parent, allow_compact_timestamp=allow_compact_timestamp
    )
    strong_model = find_best_model_checkpoint(strong_ts_dir) if strong_ts_dir else None

    return str(strong_model) if strong_model else None

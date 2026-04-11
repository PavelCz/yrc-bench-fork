from pathlib import Path
from typing import List, Optional

# Server-specific paths
SERVER_PATHS = {
    "chai": {
        "checkpoint_base": "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        "seeds_base": "/nas/ucb/czempin/data/goal-misgen/seeds/icml",
        "svdd_base": "/nas/ucb/czempin/data/goal-misgen/trained_svdd",
        "log_base": "/nas/ucb/czempin/data/goal-misgen/slurm-logs",
    },
    "snoopy": {
        "checkpoint_base": "/scr/pavel/data/goal-misgen/policy/icml",
        "seeds_base": "/scr/pavel/data/goal-misgen/seeds/icml",
        "svdd_base": "/scr/pavel/data/goal-misgen/trained_svdd",
        "log_base": "/scr/pavel/data/goal-misgen/slurm-logs",
    },
}

# Environment choices
ENVS = ["maze", "coinrun"]

# Method to config file mapping
METHOD_CONFIGS = {
    "max-prob": "max_prob.yaml",
    "max-logit": "max_logit.yaml",
    "lb-random": "level_based_random.yaml",
    "ts-random": "timestep_random.yaml",
    "svdd-image": "image_svdd.yaml",
    "svdd-latent": "latent_svdd.yaml",
    "ensemble": "ensemble_variance.yaml",
    "ensemble-single": "ensemble_variance_single.yaml",
    "wait": "wait.yaml",
}

# Method to run name suffix mapping
METHOD_NAMES = {
    "max-prob": "max_prob",
    "max-logit": "max_logit",
    "lb-random": "lb_random",
    "ts-random": "ts_random",
    "svdd-image": "svdd_image",
    "svdd-latent": "svdd_latent",
    "ensemble": "ensemble",
    "ensemble-single": "ensemble_single",
    "wait": "wait",
}

SVDD_METHODS = {"svdd-image", "svdd-latent"}
ENSEMBLE_METHODS = {"ensemble", "ensemble-single"}
DEFAULT_NUM_ENSEMBLE_MEMBERS = 4
EXPECTED_TIMESTEPS = 200015872


def get_env_folder(env: str) -> str:
    """Get the environment folder name."""
    if env == "coinrun":
        return "coinrun"
    return f"{env}_afh"


def get_svdd_feature_type(method: str) -> str:
    """Get the SVDD feature type from method name."""
    if method == "svdd-image":
        return "image"
    if method == "svdd-latent":
        return "latent"
    return ""


def find_newest_timestamp_dir(parent_dir: Path) -> Optional[Path]:
    """Find the newest timestamp directory in parent_dir."""
    if not parent_dir.exists():
        return None

    timestamp_dirs = []
    for child in parent_dir.iterdir():
        if child.is_dir() and "__seed_" in child.name:
            timestamp_dirs.append(child)

    if not timestamp_dirs:
        return None

    if len(timestamp_dirs) > 1:
        print(f"Warning: Multiple timestamp dirs in {parent_dir}, using newest:")
        for child in sorted(timestamp_dirs, key=lambda x: x.name):
            print(f"  - {child.name}")

    return sorted(timestamp_dirs, key=lambda x: x.name)[-1]


def find_best_model_checkpoint(ts_dir: Path) -> Optional[Path]:
    """Find the model checkpoint with highest timesteps."""
    if not ts_dir.exists():
        return None

    model_files = []
    for path in ts_dir.iterdir():
        if (
            path.is_file()
            and path.name.startswith("model_")
            and path.name.endswith(".pth")
        ):
            stem = path.stem
            try:
                timesteps = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            model_files.append((timesteps, path))

    if not model_files:
        return None

    model_files.sort(key=lambda x: x[0])
    highest_timesteps, best_model = model_files[-1]

    if highest_timesteps != EXPECTED_TIMESTEPS:
        print(
            f"Warning: {ts_dir.name} has max timesteps {highest_timesteps}, "
            f"expected {EXPECTED_TIMESTEPS}"
        )

    return best_model


def get_checkpoints(env: str, exp_id: int, checkpoint_base_path: str) -> dict:
    """Get checkpoint paths based on environment and experiment ID."""
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


def get_svdd_policy_name(env: str, exp_id: int, method: str) -> str:
    """Get the SVDD policy directory name."""
    feature_type = get_svdd_feature_type(method)
    return f"svdd_{env}_{feature_type}_exp{exp_id}"


def get_svdd_model_path(
    env: str, exp_id: int, method: str, svdd_base_path: str
) -> Optional[str]:
    """Get the full path to the trained SVDD model file."""
    policy_name = get_svdd_policy_name(env, exp_id, method)
    model_file = Path(svdd_base_path) / policy_name / "trained.joblib"
    if model_file.exists():
        return str(model_file)
    return None


def get_ensemble_member_paths(
    env: str,
    exp_id: int,
    checkpoint_base_path: str,
    num_members: int = DEFAULT_NUM_ENSEMBLE_MEMBERS,
) -> List[Optional[str]]:
    """Get paths to ensemble member checkpoints."""
    env_folder = get_env_folder(env)
    base_path = Path(checkpoint_base_path) / env_folder / "ensembles"

    member_paths = []
    for member_id in range(num_members):
        member_parent = base_path / f"icml2_ensemble_{env}_exp{exp_id}_m{member_id}"
        member_ts_dir = find_newest_timestamp_dir(member_parent)
        member_model = (
            find_best_model_checkpoint(member_ts_dir) if member_ts_dir else None
        )

        if member_model:
            member_paths.append(str(member_model))
        else:
            member_paths.append(None)

    return member_paths

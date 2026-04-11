import json
from pathlib import Path
from typing import Any, Dict


def default_coordination_artifact_root(checkpoint_base_path: Path) -> Path:
    """Derive a coordination-artifact root separate from acting checkpoints.

    Example:
    ``/scr/.../policy/icml`` -> ``/scr/.../coordination-policies/icml``
    """
    return (
        checkpoint_base_path.parent.parent
        / "coordination-policies"
        / checkpoint_base_path.name
    )


def resolve_coordination_artifact_dir(
    env: str,
    exp_id: int,
    method_name: str,
    run_key: str,
    *,
    coordination_root: Path,
) -> Path:
    """Return the coordination-method artifact directory for one run."""
    return coordination_root / env / f"exp{exp_id}" / method_name / run_key


def resolve_calibration_path(coordination_artifact_dir: Path) -> Path:
    """Return the canonical calibration artifact path for a coordination run."""
    return coordination_artifact_dir / "calibration.npz"


def resolve_metadata_path(coordination_artifact_dir: Path) -> Path:
    """Return the canonical metadata manifest path for a coordination run."""
    return coordination_artifact_dir / "metadata.json"


def write_coordination_metadata(
    coordination_artifact_dir: Path, metadata: Dict[str, Any]
) -> Path:
    """Write a coordination-artifact metadata manifest."""
    coordination_artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = resolve_metadata_path(coordination_artifact_dir)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata_path

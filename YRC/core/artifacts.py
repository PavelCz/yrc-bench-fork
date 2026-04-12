from pathlib import Path


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

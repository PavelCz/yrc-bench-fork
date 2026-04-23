import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


ROLLOUT_CHUNK_FORMAT = "chunked_rollouts_v1"


class RolloutChunkWriter:
    """Write rollout observations to bounded chunk files plus a JSON manifest."""

    def __init__(self, manifest_path: Path, chunks_dir: Path):
        self.manifest_path = Path(manifest_path)
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: List[Dict[str, Any]] = []
        self._num_observations = 0
        self._first_observation_shape: Optional[List[int]] = None

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_observations(self) -> int:
        return self._num_observations

    @property
    def first_observation_shape(self) -> Optional[List[int]]:
        return self._first_observation_shape

    def write_chunk(self, observations: List[torch.Tensor]) -> None:
        if not observations:
            return

        if self._first_observation_shape is None:
            self._first_observation_shape = list(observations[0].shape)

        chunk_name = f"chunk_{len(self._chunks):06d}.pt"
        chunk_path = self.chunks_dir / chunk_name
        with chunk_path.open("wb") as f:
            torch.save(list(observations), f)

        relative_chunk_path = chunk_path.relative_to(self.manifest_path.parent)
        self._chunks.append(
            {
                "path": str(relative_chunk_path),
                "num_observations": len(observations),
            }
        )
        self._num_observations += len(observations)

    def save_manifest(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        manifest = {
            "format": ROLLOUT_CHUNK_FORMAT,
            "num_chunks": self.num_chunks,
            "num_observations": self.num_observations,
            "first_observation_shape": self.first_observation_shape,
            "chunks": self._chunks,
        }
        if metadata is not None:
            manifest["metadata"] = metadata

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)


def is_rollout_chunk_manifest(path: Path) -> bool:
    path = Path(path)
    return path.suffix == ".json" and path.name.startswith("rollouts_manifest_")


def load_chunked_rollouts(
    manifest_path: Path, max_observations: Optional[int] = None
) -> List[torch.Tensor]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    if manifest.get("format") != ROLLOUT_CHUNK_FORMAT:
        raise ValueError(
            f"Expected rollout manifest format {ROLLOUT_CHUNK_FORMAT!r}, got "
            f"{manifest.get('format')!r} in {manifest_path}"
        )

    if max_observations is not None and max_observations < 0:
        raise ValueError(
            f"max_observations must be non-negative, got {max_observations}"
        )

    rollout_obs: List[torch.Tensor] = []
    for chunk in manifest["chunks"]:
        if max_observations is not None and len(rollout_obs) >= max_observations:
            break
        chunk_path = manifest_path.parent / chunk["path"]
        with chunk_path.open("rb") as f:
            chunk_obs = torch.load(f)
        if not isinstance(chunk_obs, list):
            raise ValueError(
                f"Expected rollout chunk {chunk_path} to contain a list, got "
                f"{type(chunk_obs)}"
            )
        if max_observations is not None:
            remaining = max_observations - len(rollout_obs)
            rollout_obs.extend(chunk_obs[:remaining])
        else:
            rollout_obs.extend(chunk_obs)

    expected_num_observations = manifest.get("num_observations")
    if (
        max_observations is None
        and expected_num_observations is not None
        and len(rollout_obs) != expected_num_observations
    ):
        raise ValueError(
            f"Loaded {len(rollout_obs)} observations from {manifest_path}, expected "
            f"{expected_num_observations}."
        )

    return rollout_obs

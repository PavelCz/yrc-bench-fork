import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


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


class IndexedChunkedRolloutDataset(Dataset):
    """Random-access dataset backed by rollout chunk files."""

    def __init__(
        self,
        manifest_path: Path,
        max_observations: Optional[int] = None,
        chunk_cache_size: int = 2,
    ):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r") as f:
            self.manifest = json.load(f)

        if self.manifest.get("format") != ROLLOUT_CHUNK_FORMAT:
            raise ValueError(
                f"Expected rollout manifest format {ROLLOUT_CHUNK_FORMAT!r}, got "
                f"{self.manifest.get('format')!r} in {self.manifest_path}"
            )

        if max_observations is not None and max_observations < 0:
            raise ValueError(
                f"max_observations must be non-negative, got {max_observations}"
            )
        if chunk_cache_size < 0:
            raise ValueError(
                f"chunk_cache_size must be non-negative, got {chunk_cache_size}"
            )

        self.chunks = list(self.manifest["chunks"])
        self.chunk_cache_size = chunk_cache_size
        self._chunk_cache = OrderedDict()

        self._cumulative_offsets: List[int] = []
        total_observations = 0
        for chunk in self.chunks:
            total_observations += int(chunk["num_observations"])
            self._cumulative_offsets.append(total_observations)

        manifest_total = self.manifest.get("num_observations")
        if manifest_total is not None and int(manifest_total) != total_observations:
            raise ValueError(
                f"Manifest {self.manifest_path} declares {manifest_total} "
                f"observations, but chunks sum to {total_observations}."
            )

        self._length = total_observations
        if max_observations is not None:
            self._length = min(self._length, max_observations)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        chunk_index, local_index = self.get_chunk_local_index(index)
        return self._load_chunk(chunk_index)[local_index]

    def get_chunk_local_index(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Rollout index {index} out of range for length {len(self)}"
            )

        chunk_index = bisect_right(self._cumulative_offsets, index)
        previous_offset = (
            0 if chunk_index == 0 else self._cumulative_offsets[chunk_index - 1]
        )
        return chunk_index, index - previous_offset

    def _load_chunk(self, chunk_index: int) -> List[torch.Tensor]:
        cached_chunk = self._chunk_cache.get(chunk_index)
        if cached_chunk is not None:
            self._chunk_cache.move_to_end(chunk_index)
            return cached_chunk

        chunk_path = self.manifest_path.parent / self.chunks[chunk_index]["path"]
        with chunk_path.open("rb") as f:
            chunk_obs = torch.load(f)
        if not isinstance(chunk_obs, list):
            raise ValueError(
                f"Expected rollout chunk {chunk_path} to contain a list, got "
                f"{type(chunk_obs)}"
            )

        expected_count = int(self.chunks[chunk_index]["num_observations"])
        if len(chunk_obs) != expected_count:
            raise ValueError(
                f"Loaded {len(chunk_obs)} observations from {chunk_path}, expected "
                f"{expected_count}."
            )

        if self.chunk_cache_size > 0:
            self._chunk_cache[chunk_index] = chunk_obs
            self._chunk_cache.move_to_end(chunk_index)
            while len(self._chunk_cache) > self.chunk_cache_size:
                self._chunk_cache.popitem(last=False)
        return chunk_obs


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

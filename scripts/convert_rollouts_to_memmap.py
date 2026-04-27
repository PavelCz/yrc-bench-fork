#!/usr/bin/env python3
"""Convert chunked rollout observations to a NumPy memmap dataset."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from YRC.core.rollout_storage import (
    ROLLOUT_MEMMAP_FORMAT,
    is_rollout_chunk_manifest,
)


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    if not is_rollout_chunk_manifest(manifest_path):
        raise ValueError(f"Expected a rollouts_manifest_*.json file: {manifest_path}")
    return manifest


def infer_memmap_paths(manifest_path: Path, output_dir: Path = None) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else manifest_path.parent
    suffix = manifest_path.stem[len("rollouts_manifest_") :]
    return {
        "data": output_dir / f"rollouts_memmap_{suffix}.dat",
        "metadata": output_dir / f"rollouts_memmap_{suffix}.json",
    }


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(tensor):
        raise ValueError(f"Expected rollout observation tensor, got {type(tensor)}")
    return tensor.detach().cpu().numpy()


def convert_rollouts_to_memmap(
    manifest_path: Path,
    *,
    output_dir: Path = None,
    dtype: str = None,
    overwrite: bool = False,
) -> Path:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    paths = infer_memmap_paths(manifest_path, output_dir=output_dir)
    data_path = paths["data"]
    metadata_path = paths["metadata"]

    if (data_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Memmap output already exists at {data_path} or {metadata_path}. "
            "Pass --overwrite to replace it."
        )

    first_shape = manifest.get("first_observation_shape")
    if first_shape is None:
        raise ValueError(f"Manifest {manifest_path} is missing first_observation_shape")
    num_observations = int(manifest["num_observations"])
    np_dtype = np.dtype(dtype) if dtype is not None else None

    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    memmap_array = None
    write_index = 0
    for chunk_index, chunk in enumerate(manifest["chunks"], start=1):
        chunk_path = manifest_path.parent / chunk["path"]
        with chunk_path.open("rb") as f:
            chunk_obs = torch.load(f)
        if not isinstance(chunk_obs, list):
            raise ValueError(
                f"Expected rollout chunk {chunk_path} to contain a list, got "
                f"{type(chunk_obs)}"
            )
        expected_count = int(chunk["num_observations"])
        if len(chunk_obs) != expected_count:
            raise ValueError(
                f"Loaded {len(chunk_obs)} observations from {chunk_path}, expected "
                f"{expected_count}."
            )
        if not chunk_obs:
            continue

        chunk_array = np.stack([tensor_to_numpy(obs) for obs in chunk_obs], axis=0)
        if np_dtype is None:
            np_dtype = chunk_array.dtype
        if list(chunk_array.shape[1:]) != list(first_shape):
            raise ValueError(
                f"Chunk {chunk_path} has observation shape {chunk_array.shape[1:]}, "
                f"expected {first_shape}."
            )

        if memmap_array is None:
            memmap_array = np.memmap(
                data_path,
                dtype=np_dtype,
                mode="w+",
                shape=(num_observations, *first_shape),
            )

        if chunk_array.dtype != np_dtype:
            chunk_array = chunk_array.astype(np_dtype, copy=False)
        next_index = write_index + len(chunk_array)
        memmap_array[write_index:next_index] = chunk_array
        write_index = next_index
        print(
            f"Converted chunk {chunk_index}/{len(manifest['chunks'])}: "
            f"{write_index}/{num_observations} observations",
            flush=True,
        )

    if memmap_array is None:
        raise ValueError(f"Manifest {manifest_path} contains no rollout observations")
    if write_index != num_observations:
        raise ValueError(
            f"Wrote {write_index} observations, expected {num_observations}."
        )
    memmap_array.flush()

    metadata = {
        "format": ROLLOUT_MEMMAP_FORMAT,
        "source_manifest": str(manifest_path),
        "data_path": data_path.name,
        "dtype": str(np_dtype),
        "shape": [num_observations, *first_shape],
        "num_observations": num_observations,
        "first_observation_shape": first_shape,
        "metadata": manifest.get("metadata", {}),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Saved memmap data to {data_path}")
    print(f"Saved memmap metadata to {metadata_path}")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert rollouts_manifest_*.json chunked rollouts to memmap."
    )
    parser.add_argument("manifest", type=Path, help="Path to rollouts manifest JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for memmap outputs. Defaults to manifest directory.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional dtype for stored observations, e.g. uint8. Defaults to chunk dtype.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing memmap outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    convert_rollouts_to_memmap(
        args.manifest,
        output_dir=args.output_dir,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

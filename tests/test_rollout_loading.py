import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from YRC.core.rollout_storage import RolloutChunkWriter
from YRC.core.rollout_storage import IndexedChunkedRolloutDataset
from YRC.core.rollout_storage import MemmapRolloutDataset
from YRC.core.utils import load_rollout_dataset_from_file, load_rollouts_from_file
from scripts.convert_rollouts_to_memmap import convert_rollouts_to_memmap


def test_load_rollouts_from_specific_file():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        rollout_file = rollout_dir / "rollouts_3levels.pt"
        config_file = rollout_dir / "rollouts_config_3levels.json"

        torch.save([torch.tensor([1.0]), torch.tensor([2.0])], rollout_file)
        config_file.write_text(json.dumps({"name": "dummy"}))

        rollout_obs = load_rollouts_from_file(rollout_file)

    assert [tensor.item() for tensor in rollout_obs] == [1.0, 2.0]


def test_load_rollouts_from_chunked_manifest_file():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk([torch.tensor([1.0]), torch.tensor([2.0])])
        writer.write_chunk([torch.tensor([3.0])])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))

        rollout_obs = load_rollouts_from_file(manifest_file)

    assert [tensor.item() for tensor in rollout_obs] == [1.0, 2.0, 3.0]


def test_indexed_chunked_dataset_maps_global_indices_to_chunks():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk([torch.tensor([1.0]), torch.tensor([2.0])])
        writer.write_chunk([torch.tensor([3.0]), torch.tensor([4.0])])
        writer.write_chunk([torch.tensor([5.0])])
        writer.save_manifest()

        dataset = IndexedChunkedRolloutDataset(manifest_file)

    assert len(dataset) == 5
    assert dataset.get_chunk_local_index(0) == (0, 0)
    assert dataset.get_chunk_local_index(1) == (0, 1)
    assert dataset.get_chunk_local_index(2) == (1, 0)
    assert dataset.get_chunk_local_index(4) == (2, 0)


def test_indexed_chunked_dataset_random_access_matches_eager_loading():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk([torch.tensor([1.0]), torch.tensor([2.0])])
        writer.write_chunk([torch.tensor([3.0])])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))

        eager_rollouts = load_rollouts_from_file(manifest_file)
        dataset = IndexedChunkedRolloutDataset(manifest_file)

        for index in [2, 0, 1]:
            assert torch.equal(dataset[index], eager_rollouts[index])


def test_load_rollouts_from_directory_with_single_chunked_dataset():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk([torch.tensor([1.0])])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))

        rollout_obs = load_rollouts_from_file(rollout_dir)

    assert [tensor.item() for tensor in rollout_obs] == [1.0]


def test_load_rollouts_prefers_largest_dataset_when_requested():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        torch.save([torch.tensor([1.0])], rollout_dir / "rollouts_64levels.pt")
        torch.save(
            [torch.tensor([1.0]), torch.tensor([2.0])],
            rollout_dir / "rollouts_128levels.pt",
        )
        (rollout_dir / "rollouts_config_64levels.json").write_text(
            json.dumps({"name": "small"})
        )
        (rollout_dir / "rollouts_config_128levels.json").write_text(
            json.dumps({"name": "large"})
        )

        rollout_obs = load_rollouts_from_file(rollout_dir, prefer_largest=True)

    assert [tensor.item() for tensor in rollout_obs] == [1.0, 2.0]


def test_load_rollouts_limits_chunked_dataset_by_completed_levels():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        metadata_file = rollout_dir / "rollouts_metadata_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk(
            [
                torch.tensor([1.0]),
                torch.tensor([2.0]),
                torch.tensor([3.0]),
                torch.tensor([4.0]),
                torch.tensor([5.0]),
                torch.tensor([6.0]),
            ]
        )
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))
        metadata_file.write_text(
            json.dumps({"completed_rollout_observation_counts": [2, 3, 1]})
        )

        rollout_obs = load_rollouts_from_file(rollout_dir, max_levels=2)

    assert [tensor.item() for tensor in rollout_obs] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_indexed_chunked_dataset_limits_length_by_completed_levels():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        metadata_file = rollout_dir / "rollouts_metadata_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk([torch.tensor([1.0]), torch.tensor([2.0])])
        writer.write_chunk([torch.tensor([3.0]), torch.tensor([4.0])])
        writer.write_chunk([torch.tensor([5.0]), torch.tensor([6.0])])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))
        metadata_file.write_text(
            json.dumps({"completed_rollout_observation_counts": [2, 3, 1]})
        )

        dataset = load_rollout_dataset_from_file(
            rollout_dir,
            max_levels=2,
            streaming_rollouts="true",
        )

        assert isinstance(dataset, IndexedChunkedRolloutDataset)
        assert len(dataset) == 5
        assert [dataset[index].item() for index in range(len(dataset))] == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]


def test_indexed_chunked_dataset_lru_cache_evicts_after_configured_chunk_count():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )

        writer.write_chunk([torch.tensor([1.0])])
        writer.write_chunk([torch.tensor([2.0])])
        writer.write_chunk([torch.tensor([3.0])])
        writer.save_manifest()

        dataset = IndexedChunkedRolloutDataset(manifest_file, chunk_cache_size=2)
        assert dataset[0].item() == 1.0
        assert dataset[1].item() == 2.0
        assert dataset[2].item() == 3.0

    assert list(dataset._chunk_cache.keys()) == [1, 2]


def test_streaming_loader_keeps_legacy_pt_rollouts_eager():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        rollout_file = rollout_dir / "rollouts_3levels.pt"
        config_file = rollout_dir / "rollouts_config_3levels.json"

        torch.save([torch.tensor([1.0]), torch.tensor([2.0])], rollout_file)
        config_file.write_text(json.dumps({"name": "dummy"}))

        rollout_obs = load_rollout_dataset_from_file(
            rollout_file,
            streaming_rollouts="true",
        )

    assert isinstance(rollout_obs, list)
    assert [tensor.item() for tensor in rollout_obs] == [1.0, 2.0]


def test_convert_chunked_rollouts_to_memmap_and_random_access():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )
        writer.write_chunk(
            [
                torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
                torch.tensor([[5, 6], [7, 8]], dtype=torch.uint8),
            ]
        )
        writer.write_chunk([torch.tensor([[9, 10], [11, 12]], dtype=torch.uint8)])
        writer.save_manifest({"completed_rollout_observation_counts": [1, 1, 1]})

        metadata_path = convert_rollouts_to_memmap(manifest_file)
        dataset = MemmapRolloutDataset(metadata_path)

    assert len(dataset) == 3
    assert dataset[0].dtype == torch.uint8
    assert torch.equal(dataset[2], torch.tensor([[9, 10], [11, 12]], dtype=torch.uint8))


def test_streaming_loader_prefers_memmap_for_matching_level_count():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )
        writer.write_chunk([torch.tensor([1], dtype=torch.uint8)])
        writer.write_chunk([torch.tensor([2], dtype=torch.uint8)])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))
        convert_rollouts_to_memmap(manifest_file)

        dataset = load_rollout_dataset_from_file(
            rollout_dir,
            prefer_largest=True,
            streaming_rollouts="auto",
        )

    assert isinstance(dataset, MemmapRolloutDataset)
    assert [dataset[index].item() for index in range(len(dataset))] == [1, 2]


def test_loader_require_memmap_rejects_chunked_rollout_dataset():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )
        writer.write_chunk([torch.tensor([1], dtype=torch.uint8)])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))

        try:
            load_rollout_dataset_from_file(
                rollout_dir,
                prefer_largest=True,
                streaming_rollouts="auto",
                require_memmap=True,
            )
        except ValueError as exc:
            error = str(exc)
        else:
            raise AssertionError("Expected require_memmap=True to reject chunked data")

    assert "SVDD rollout training requires a memmap rollout artifact" in error
    assert "convert_rollouts_to_memmap.py" in error


def test_loader_require_memmap_accepts_memmap_rollout_dataset():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )
        writer.write_chunk([torch.tensor([1], dtype=torch.uint8)])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))
        convert_rollouts_to_memmap(manifest_file)

        dataset = load_rollout_dataset_from_file(
            rollout_dir,
            prefer_largest=True,
            streaming_rollouts="auto",
            require_memmap=True,
        )

    assert isinstance(dataset, MemmapRolloutDataset)
    assert dataset[0].item() == 1


def test_memmap_dataset_limits_length_by_completed_levels():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_3levels.json"
        config_file = rollout_dir / "rollouts_config_3levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_3levels_chunks"
        )
        writer.write_chunk([torch.tensor([1]), torch.tensor([2])])
        writer.write_chunk([torch.tensor([3]), torch.tensor([4])])
        writer.write_chunk([torch.tensor([5]), torch.tensor([6])])
        writer.save_manifest({"completed_rollout_observation_counts": [2, 3, 1]})
        config_file.write_text(json.dumps({"name": "dummy"}))
        convert_rollouts_to_memmap(manifest_file)

        dataset = load_rollout_dataset_from_file(
            rollout_dir,
            max_levels=2,
            prefer_largest=True,
            streaming_rollouts="auto",
        )

    assert isinstance(dataset, MemmapRolloutDataset)
    assert len(dataset) == 5
    assert [dataset[index].item() for index in range(len(dataset))] == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_loader_rejects_large_non_memmap_rollout_dataset():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_1025levels.json"
        config_file = rollout_dir / "rollouts_config_1025levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_1025levels_chunks"
        )
        writer.write_chunk([torch.tensor([1.0])])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))

        try:
            load_rollout_dataset_from_file(
                rollout_dir,
                prefer_largest=True,
                streaming_rollouts="auto",
            )
        except ValueError as exc:
            error = str(exc)
        else:
            raise AssertionError("Expected large non-memmap rollout dataset to fail")

    assert "requires a memmap rollout artifact" in error


def test_loader_allows_large_artifact_when_rollout_max_levels_is_small():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        manifest_file = rollout_dir / "rollouts_manifest_1025levels.json"
        config_file = rollout_dir / "rollouts_config_1025levels.json"
        metadata_file = rollout_dir / "rollouts_metadata_1025levels.json"
        writer = RolloutChunkWriter(
            manifest_file, rollout_dir / "rollouts_1025levels_chunks"
        )
        writer.write_chunk([torch.tensor([1.0]), torch.tensor([2.0])])
        writer.save_manifest()
        config_file.write_text(json.dumps({"name": "dummy"}))
        metadata_file.write_text(
            json.dumps({"completed_rollout_observation_counts": [1] * 1025})
        )

        dataset = load_rollout_dataset_from_file(
            rollout_dir,
            max_levels=1024,
            prefer_largest=True,
            streaming_rollouts="auto",
        )

    assert isinstance(dataset, IndexedChunkedRolloutDataset)

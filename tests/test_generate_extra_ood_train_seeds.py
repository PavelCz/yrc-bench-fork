import json
import sys
from pathlib import Path


def import_seed_generator():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import generate_extra_ood_train_seeds
    finally:
        sys.path.pop(0)
    return generate_extra_ood_train_seeds


def write_seed_file(path: Path, seed_splits: dict) -> None:
    path.write_text(json.dumps({"metadata": {}, "seeds": seed_splits}))


def test_generate_extra_ood_train_seeds_excludes_all_existing_splits(tmp_path):
    generator = import_seed_generator()
    seed_file = tmp_path / "0.json"
    write_seed_file(
        seed_file,
        {
            "policy_train": [1, 2, 3],
            "ood_train": [100_000, 100_010],
            "validation": [100_020],
            "ood_eval": [100_030],
        },
    )

    source_files = generator.discover_level_seed_files([seed_file])
    excluded_seeds = generator.load_excluded_seeds(source_files)
    generated_seeds = generator.generate_ood_train_seeds(
        20,
        excluded_seeds,
        base_seed=7,
        min_seed=100_000,
        max_seed=100_100,
    )

    assert len(generated_seeds) == 20
    assert len(set(generated_seeds)) == 20
    assert set(generated_seeds).isdisjoint(excluded_seeds)
    assert all(100_000 <= seed <= 100_100 for seed in generated_seeds)


def test_build_seed_file_data_writes_only_ood_train_split(tmp_path):
    generator = import_seed_generator()
    source_file = tmp_path / "0.json"
    generated_seeds = [100_101, 100_102, 100_103]

    seed_file_data = generator.build_seed_file_data(
        generated_seeds,
        source_files=[source_file],
        excluded_count=10,
        base_seed=6033,
        min_seed=100_000,
        max_seed=200_000,
    )

    assert seed_file_data["seeds"]["policy_train"] == []
    assert seed_file_data["seeds"]["ood_train"] == generated_seeds
    assert seed_file_data["seeds"]["validation"] == []
    assert seed_file_data["seeds"]["ood_eval"] == []
    assert seed_file_data["metadata"]["num_ood_train"] == 3

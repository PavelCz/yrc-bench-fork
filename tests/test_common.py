import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

common = importlib.import_module("common")
resolve_acting_checkpoint = importlib.import_module("resolve_acting_checkpoint")


def _make_run(run_dir: Path, seed: int = 4, steps: int = common.EXPECTED_TIMESTEPS):
    ts_dir = run_dir / f"2026-09-14__02-50-05__seed_{seed}"
    ts_dir.mkdir(parents=True)
    checkpoint = ts_dir / f"model_{steps}.pth"
    checkpoint.write_text("ckpt")
    return checkpoint


def test_get_heist400_strong_checkpoint_uses_sibling_policy_tree(tmp_path):
    icml_base = tmp_path / "policy" / "icml"
    run_dir = tmp_path / "policy" / "heist400" / "heist_afh" / "heist400_heist_exp4_50p"
    checkpoint = _make_run(run_dir, seed=4, steps=common.HEIST400_CHECKPOINT_STEPS)

    path = common.get_heist400_strong_checkpoint(4, str(icml_base))
    assert path == str(checkpoint)


def test_get_checkpoint_at_steps_returns_expected_filename_before_file_exists(
    tmp_path,
):
    run_dir = tmp_path / "icml2_heist_exp4_0p"
    ts_dir = run_dir / "2026-09-14__02-50-05__seed_4"
    ts_dir.mkdir(parents=True)
    # Partial checkpoint only.
    (ts_dir / "model_40042496.pth").write_text("partial")

    path = common.get_checkpoint_at_steps(run_dir, common.EXPECTED_TIMESTEPS)
    assert path == str(ts_dir / f"model_{common.EXPECTED_TIMESTEPS}.pth")
    assert not Path(path).exists()


def test_resolve_acting_checkpoint_prints_heist400_path(tmp_path, capsys, monkeypatch):
    icml_base = tmp_path / "policy" / "icml"
    run_dir = tmp_path / "policy" / "heist400" / "heist_afh" / "heist400_heist_exp4_50p"
    checkpoint = _make_run(run_dir, seed=4, steps=common.HEIST400_CHECKPOINT_STEPS)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resolve_acting_checkpoint.py",
            "--checkpoint-base",
            str(icml_base),
            "--env",
            "heist",
            "--exp-id",
            "4",
            "--heist400",
        ],
    )
    assert resolve_acting_checkpoint.main() == 0
    assert capsys.readouterr().out.strip() == str(checkpoint)


def test_resolve_acting_checkpoint_errors_if_final_file_missing(tmp_path, monkeypatch):
    icml_base = tmp_path / "policy" / "icml"
    run_dir = icml_base / "heist_afh" / "icml2_heist_exp4_0p"
    ts_dir = run_dir / "2026-09-14__02-50-05__seed_4"
    ts_dir.mkdir(parents=True)
    (ts_dir / "model_40042496.pth").write_text("partial")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resolve_acting_checkpoint.py",
            "--checkpoint-base",
            str(icml_base),
            "--env",
            "heist",
            "--exp-id",
            "4",
            "--percent",
            "0",
        ],
    )
    assert resolve_acting_checkpoint.main() == 1


def test_build_sbatch_cli_adds_dependency_and_kill_on_invalid_dep():
    assert common.build_sbatch_cli() == ["sbatch"]
    assert common.build_sbatch_cli(dependency="afterok:1:2") == [
        "sbatch",
        "--dependency=afterok:1:2",
        "--kill-on-invalid-dep=yes",
    ]
    assert common.build_sbatch_cli(dependency="afterok:1", extra_args=["--nice=0"]) == [
        "sbatch",
        "--dependency=afterok:1",
        "--kill-on-invalid-dep=yes",
        "--nice=0",
    ]

import importlib
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

run_eval = importlib.import_module("run_eval")


def test_coinrun_proxy_fail_uses_coinrun_artifacts():
    assert run_eval.ARTIFACT_ENVS["coinrun_proxy_fail"] == "coinrun"
    assert "coinrun_proxy_fail" in run_eval.EVAL_ENVS


def test_eval_sbatch_overrides_env_name():
    command = run_eval.build_sbatch_command(
        "job",
        {
            "config": "configs/eval/coinrun/max_prob.yaml",
            "name": "name",
            "env_name": "coinrun_proxy_fail",
            "experiment_group": "group",
            "video_episodes_to_collect": 0,
            "num_levels": 16,
            "video_filter": "all",
            "cp_rolling_average": "none",
            "video_logging_mode": "none",
            "video_filter_mode": "any",
            "sim": "sim.pth",
            "weak": "weak.pth",
            "strong": "strong.pth",
            "level_seeds_file": "seeds.json",
            "coverage_fraction": 0.05,
        },
        "ood-stable",
        Path("/tmp/logs"),
    )

    assert "-en coinrun_proxy_fail" in command
    assert "-weak weak.pth" in command


def test_svdd_model_path_uses_training_prefix(tmp_path):
    model_file = (
        tmp_path
        / "neurips02"
        / "svdd_coinrun_image_exp0"
        / "trained.joblib"
    )
    model_file.parent.mkdir(parents=True)
    model_file.write_text("model")

    resolved = run_eval.get_svdd_model_path(
        "coinrun", 0, "svdd-image", str(tmp_path), "neurips02"
    )

    assert resolved == str(model_file)
    assert run_eval.get_svdd_expected_model_path(
        "coinrun", 0, "svdd-image", str(tmp_path), "neurips02"
    ) == model_file


def test_preflight_uses_module_invocation_and_hides_success_output(
    monkeypatch, capsys
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "preflight ok\n", "")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval.run_preflight_check(
        "ood-stable", "coinrun_proxy_fail", show_output=False
    )

    assert calls[0][0] == [
        "conda",
        "run",
        "-n",
        "ood-stable",
        "python",
        "-m",
        "scripts.preflight_eval_env",
        "--env",
        "coinrun_proxy_fail",
    ]
    assert calls[0][1]["cwd"] == run_eval.REPO_ROOT
    assert capsys.readouterr().out == ""


def test_preflight_prints_output_for_dry_run(monkeypatch, capsys):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "preflight ok\n", "")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval.run_preflight_check(
        "ood-stable", "coinrun_proxy_fail", show_output=True
    )

    output = capsys.readouterr().out
    assert "=== Preflight check ===" in output
    assert "python -m scripts.preflight_eval_env --env coinrun_proxy_fail" in output
    assert "preflight ok" in output


def test_main_stops_before_submit_when_preflight_fails(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--prefix",
            "prefix",
            "--exp-ids",
            "1",
            "--server",
            "chai",
            "--env",
            "coinrun_proxy_fail",
            "--method",
            "max-logit",
            "--conda-env",
            "ood-stable",
        ],
    )
    monkeypatch.setattr(run_eval, "run_preflight_check", lambda *args, **kwargs: False)

    def fail_submit(*args, **kwargs):
        raise AssertionError("submit_job should not be called after preflight failure")

    monkeypatch.setattr(run_eval, "submit_job", fail_submit)

    assert run_eval.main() == 1

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


def make_eval_args(name="name"):
    return {
        "config": "configs/eval/coinrun/max_prob.yaml",
        "name": name,
        "env_name": "coinrun_proxy_fail",
        "experiment_group": f"group_{name}",
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
        "calibration_levels": None,
    }


def make_job_spec(exp_id):
    job_name = f"coinrun_max-prob_exp{exp_id}"
    return {
        "job_name": job_name,
        "eval_args": make_eval_args(job_name),
        "exp_id": exp_id,
        "env": "coinrun",
        "method": "max-prob",
        "robust_checkpoint_key": None,
    }


def test_packed_sbatch_runs_multiple_eval_steps_on_one_gpu():
    command = run_eval.build_packed_sbatch_command(
        "coinrun_max-prob_exp0-3",
        [make_job_spec(exp_id) for exp_id in range(4)],
        "ood-stable",
        Path("/tmp/logs"),
    )

    assert "#SBATCH --gres=gpu:1" in command
    assert "#SBATCH --nodes=1" in command
    assert "#SBATCH --ntasks=4" in command
    assert "#SBATCH --cpus-per-task=30" in command
    assert "#SBATCH --mem=400G" in command
    assert command.count("srun --overlap --ntasks=1 --cpus-per-task=30") == 4
    assert command.count("--gres=gpu:1") == 5
    assert command.count("python eval_afhp.py") == 4
    assert 'pids+=("$!")' in command
    assert "-n coinrun_max-prob_exp0" in command
    assert "-n coinrun_max-prob_exp3" in command


def test_container_sbatch_wraps_eval_in_apptainer():
    command = run_eval.build_sbatch_command(
        "job",
        make_eval_args("job"),
        "ood-stable",
        Path("/tmp/logs"),
        execution="apptainer",
        container_image=Path("/repo/yrc-bench-procgen.sif"),
        repo_dir=Path("/repo"),
        container_binds=["/path/to/cluster3:/path/to/cluster3"],
    )

    assert "conda activate" not in command
    assert "module load apptainer" in command
    assert "apptainer exec --nv" in command
    assert "--bind /repo:/workspace" in command
    assert "--bind /path/to/cluster3:/path/to/cluster3" in command
    assert "--pwd /workspace /repo/yrc-bench-procgen.sif bash -lc" in command
    assert "Container Python not found: /opt/venv/bin/python" in command
    assert "export PATH=/opt/venv/bin:${PATH}; hash -r; python eval_afhp.py" in command
    assert "python eval_afhp.py" in command
    assert "-n job" in command


def test_eval_sbatch_passes_calibration_level_override():
    eval_args = make_eval_args("job")
    eval_args["calibration_levels"] = 64

    command = run_eval.build_sbatch_command(
        "job",
        eval_args,
        "ood-stable",
        Path("/tmp/logs"),
    )

    assert "-calibration_levels 64" in command


def test_container_packed_sbatch_wraps_each_eval_in_apptainer():
    command = run_eval.build_packed_sbatch_command(
        "coinrun_max-prob_exp0-3",
        [make_job_spec(exp_id) for exp_id in range(4)],
        "ood-stable",
        Path("/tmp/logs"),
        execution="apptainer",
        container_image=Path("/repo/yrc-bench-procgen.sif"),
        repo_dir=Path("/repo"),
        container_binds=["/path/to/cluster3:/path/to/cluster3"],
    )

    assert "conda activate" not in command
    assert command.count("srun --overlap --ntasks=1 --cpus-per-task=30") == 4
    assert command.count("apptainer exec --nv") == 4
    assert command.count("/repo/yrc-bench-procgen.sif bash -lc") == 4
    assert command.count("python eval_afhp.py") == 4


def test_container_preflight_does_not_require_gpu_visibility():
    command = run_eval.build_preflight_command(
        "ood-stable",
        "coinrun_proxy_fail",
        execution="apptainer",
        container_image=Path("/repo/yrc-bench-procgen.sif"),
        repo_dir=Path("/repo"),
        container_binds=["/path/to/cluster3:/path/to/cluster3"],
    )

    shell_command = command[-1]
    assert command[:2] == ["bash", "-lc"]
    assert "apptainer exec" in shell_command
    assert "--nv" not in shell_command
    assert "Container Python not found: /opt/venv/bin/python" in shell_command
    assert (
        "export PATH=/opt/venv/bin:${PATH}; hash -r; "
        "python -m scripts.preflight_eval_env" in shell_command
    )
    assert (
        "python -m scripts.preflight_eval_env --env coinrun_proxy_fail "
        "--skip-local-path-check" in shell_command
    )


def test_chunk_job_specs_handles_non_multiple_counts():
    chunks = run_eval.chunk_job_specs([make_job_spec(i) for i in range(5)], 4)

    assert [[spec["exp_id"] for spec in chunk] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4],
    ]


def test_svdd_model_path_uses_training_prefix(tmp_path):
    model_file = tmp_path / "dummy02" / "svdd_coinrun_image_exp0" / "trained.joblib"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("model")

    resolved = run_eval.get_svdd_model_path(
        "coinrun", 0, "svdd-image", str(tmp_path), "dummy02"
    )

    assert resolved == str(model_file)
    assert (
        run_eval.get_svdd_expected_model_path(
            "coinrun", 0, "svdd-image", str(tmp_path), "dummy02"
        )
        == model_file
    )


def test_svdd_default_prefix_is_dummy05():
    expected = run_eval.get_svdd_expected_model_path(
        "coinrun",
        0,
        "svdd-image",
        "/path/to/cluster1/data/goal-misgen/trained_svdd",
        run_eval.EVAL_DEFAULTS["svdd_prefix"],
    )

    assert run_eval.EVAL_DEFAULTS["svdd_prefix"] == "dummy05"
    assert expected == Path(
        "/path/to/cluster1/data/goal-misgen/trained_svdd/"
        "dummy05/svdd_coinrun_image_exp0/trained.joblib"
    )


def test_preflight_uses_module_invocation_and_hides_success_output(monkeypatch, capsys):
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
            "cluster1",
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


def test_main_packs_valid_exp_ids_into_gpu_chunks(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_text("model")
    seeds_base = tmp_path / "seeds"
    seeds_base.mkdir()
    for exp_id in range(5):
        (seeds_base / f"{exp_id}.json").write_text("{}")

    server_paths = run_eval.SERVER_PATHS.copy()
    server_paths["unit"] = {
        "checkpoint_base": str(tmp_path / "checkpoints"),
        "rollouts_base": str(tmp_path / "rollouts"),
        "seeds_base": str(seeds_base),
        "svdd_base": str(tmp_path / "svdd"),
        "log_base": str(tmp_path / "logs"),
        "evals_base": str(tmp_path / "evals"),
    }
    monkeypatch.setattr(run_eval, "SERVER_PATHS", server_paths)
    monkeypatch.setattr(run_eval, "run_preflight_check", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        run_eval,
        "get_checkpoints",
        lambda *args, **kwargs: {
            "sim": str(checkpoint),
            "weak": str(checkpoint),
            "strong": str(checkpoint),
        },
    )

    submitted = []

    def fake_submit_packed_job(
        job_name, job_specs, conda_env, log_dir, qos="default", dry_run=False, **kwargs
    ):
        submitted.append((job_name, [spec["exp_id"] for spec in job_specs], dry_run))

    monkeypatch.setattr(run_eval, "submit_packed_job", fake_submit_packed_job)
    monkeypatch.setattr(
        run_eval,
        "submit_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("single-job submit should not be used")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--prefix",
            "prefix",
            "--exp-ids",
            "0",
            "1",
            "2",
            "3",
            "4",
            "--server",
            "unit",
            "--env",
            "coinrun",
            "--method",
            "max-prob",
            "--runs-per-gpu",
            "4",
        ],
    )

    assert run_eval.main() == 0
    assert submitted == [
        ("coinrun_max-prob_exp0-1-2-3", [0, 1, 2, 3], False),
        ("coinrun_max-prob_exp4", [4], False),
    ]

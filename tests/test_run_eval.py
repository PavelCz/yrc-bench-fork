import importlib
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

run_eval = importlib.import_module("run_eval")
preflight_eval_env = importlib.import_module("preflight_eval_env")
sync_eval_artifacts = importlib.import_module("sync_eval_artifacts")


def test_coinrun_proxy_fail_uses_coinrun_artifacts():
    assert run_eval.ARTIFACT_ENVS["coinrun_proxy_fail"] == "coinrun"
    assert "coinrun_proxy_fail" in run_eval.EVAL_ENVS


def test_heist_proxy_fail_uses_heist_artifacts():
    assert run_eval.ARTIFACT_ENVS["heist_proxy_fail"] == "heist"
    assert "heist_proxy_fail" in run_eval.EVAL_ENVS


def test_lookup_eval_checkpoints_heist400_uses_expected_final_files(tmp_path):
    icml_base = tmp_path / "policy" / "icml"
    weak_run = icml_base / "heist_afh" / "icml2_heist_exp4_0p"
    ts_dir = weak_run / "2026-09-14__02-50-05__seed_4"
    ts_dir.mkdir(parents=True)
    (ts_dir / "model_40042496.pth").write_text("partial")
    strong_run = (
        tmp_path / "policy" / "heist400" / "heist_afh" / "heist400_heist_exp4_50p"
    )
    strong_ts = strong_run / "2026-09-13__15-15-24__seed_4"
    strong_ts.mkdir(parents=True)

    checkpoints = run_eval.lookup_eval_checkpoints(
        "heist", 4, str(icml_base), heist400=True
    )
    assert checkpoints["weak"].endswith(f"model_{run_eval.EXPECTED_TIMESTEPS}.pth")
    assert checkpoints["strong"].endswith("model_400031744.pth")


def test_lookup_eval_checkpoints_allow_missing_emits_runtime_lookup():
    checkpoints = run_eval.lookup_eval_checkpoints(
        "heist",
        4,
        "/nas/ucb/czempin/data/goal-misgen/policy/icml",
        heist400=True,
        allow_missing_checkpoints=True,
    )
    assert run_eval.is_runtime_checkpoint_lookup(checkpoints["weak"])
    assert "--percent 0" in checkpoints["weak"]
    assert run_eval.is_runtime_checkpoint_lookup(checkpoints["strong"])
    assert "--heist400" in checkpoints["strong"]
    block = run_eval.build_conda_setup_block("ood-stable")
    assert 'source "$CONDA_BASE/etc/profile.d/conda.sh"' in block
    assert "conda activate ood-stable" in block
    assert 'echo "Using conda env: ood-stable"' in block


def test_chai_cache_env_block_is_chai_only():
    chai = run_eval.build_chai_cache_env_block("chai")
    assert "export MPLCONFIGDIR=" in chai
    assert "export XDG_CACHE_HOME=" in chai
    assert "export TMPDIR=" in chai
    assert "export TEMP=" in chai
    assert "export TMP=" in chai
    assert "export CUDA_CACHE_PATH=" in chai
    assert "export CCACHE_DIR=" in chai
    assert "/nas/ttl=60d/czempin/tmp" in chai
    assert run_eval.build_chai_cache_env_block("snoopy") == ""
    assert run_eval.build_chai_cache_env_block("carc") == ""


def test_proxy_penalty_envs_use_base_artifacts_and_pass_preflight():
    expected_artifacts = {
        "coinrun_proxy_penalty": "coinrun",
        "maze_proxy_penalty": "maze",
        "heist_proxy_penalty": "heist",
    }

    for env_name, artifact_env in expected_artifacts.items():
        assert run_eval.ARTIFACT_ENVS[env_name] == artifact_env
        assert sync_eval_artifacts.ARTIFACT_ENVS[env_name] == artifact_env
        assert env_name in run_eval.EVAL_ENVS
        assert env_name in sync_eval_artifacts.EVAL_ENVS
        assert env_name in preflight_eval_env.SUPPORTED_ENVS


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
    assert 'source "$CONDA_BASE/etc/profile.d/conda.sh"' in command
    assert "conda activate ood-stable" in command
    assert "export MPLCONFIGDIR='/nas/ttl=60d/czempin/mpl-config'" in command
    assert "export XDG_CACHE_HOME='/nas/ttl=60d/czempin/xdg-cache'" in command
    assert 'export TMPDIR="/nas/ttl=60d/czempin/tmp/${SLURM_JOB_ID:-$$}"' in command
    assert "export CUDA_CACHE_PATH='/nas/ttl=60d/czempin/cuda-cache'" in command

    snoopy = run_eval.build_sbatch_command(
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
        server="snoopy",
    )
    assert "MPLCONFIGDIR" not in snoopy


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


def test_packed_sbatch_with_runtime_lookup_keeps_command_substitution():
    eval_args = make_eval_args("job")
    eval_args["sim"] = (
        '"$(python scripts/resolve_acting_checkpoint.py --env heist --exp-id 4 --percent 0)"'
    )
    eval_args["weak"] = eval_args["sim"]
    eval_args["strong"] = (
        '"$(python scripts/resolve_acting_checkpoint.py --env heist --exp-id 4 --heist400)"'
    )
    command = run_eval.build_sbatch_command(
        "job",
        eval_args,
        "ood-stable",
        Path("/tmp/logs"),
    )
    assert "--heist400" in command
    assert "resolve_acting_checkpoint.py" in command
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
    assert 'source "$CONDA_BASE/etc/profile.d/conda.sh"' in command
    assert "export MPLCONFIGDIR='/nas/ttl=60d/czempin/mpl-config'" in command
    assert 'export TMPDIR="/nas/ttl=60d/czempin/tmp/${SLURM_JOB_ID:-$$}"' in command
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
        container_binds=["/project2/biyik_1165:/project2/biyik_1165"],
    )

    assert "conda activate" not in command
    assert "module load apptainer" in command
    assert "apptainer exec --nv" in command
    assert "--bind /repo:/workspace" in command
    assert "--bind /project2/biyik_1165:/project2/biyik_1165" in command
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
        container_binds=["/project2/biyik_1165:/project2/biyik_1165"],
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
        container_binds=["/project2/biyik_1165:/project2/biyik_1165"],
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
    model_file = tmp_path / "neurips02" / "svdd_coinrun_image_exp0" / "trained.joblib"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("model")

    resolved = run_eval.get_svdd_model_path(
        "coinrun", 0, "svdd-image", str(tmp_path), "neurips02"
    )

    assert resolved == str(model_file)
    assert (
        run_eval.get_svdd_expected_model_path(
            "coinrun", 0, "svdd-image", str(tmp_path), "neurips02"
        )
        == model_file
    )


def test_svdd_default_prefix_is_neurips05():
    expected = run_eval.get_svdd_expected_model_path(
        "coinrun",
        0,
        "svdd-image",
        "/nas/ucb/czempin/data/goal-misgen/trained_svdd",
        run_eval.EVAL_DEFAULTS["svdd_prefix"],
    )

    assert run_eval.EVAL_DEFAULTS["svdd_prefix"] == "neurips05"
    assert expected == Path(
        "/nas/ucb/czempin/data/goal-misgen/trained_svdd/"
        "neurips05/svdd_coinrun_image_exp0/trained.joblib"
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


def test_train_ensemble_policies_sources_conda_sh():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_ensemble_policies.sh"
    )
    text = script.read_text()
    assert 'CONDA_BASE="/nas/ucb/czempin/anaconda3"' in text
    assert (
        ". ${CONDA_BASE}/etc/profile.d/conda.sh && cd $TRAIN_DIR && conda run" in text
    )

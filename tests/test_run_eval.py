import importlib
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

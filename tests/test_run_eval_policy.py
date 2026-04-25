import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

run_eval_policy = importlib.import_module("run_eval_policy")


def test_coinrun_proxy_fail_uses_coinrun_config_and_checkpoints():
    assert (
        run_eval_policy.DEFAULT_CONFIGS["coinrun_proxy_fail"]
        == "configs/eval/coinrun/max_prob.yaml"
    )
    assert run_eval_policy.CHECKPOINT_ENVS["coinrun_proxy_fail"] == "coinrun"
    assert "coinrun_proxy_fail" in run_eval_policy.EVAL_ENVS


def test_eval_policy_sbatch_overrides_env_name():
    command = run_eval_policy.build_sbatch_command(
        "job",
        {
            "config": "configs/eval/coinrun/max_prob.yaml",
            "name": "name",
            "experiment_group": "group",
            "eval_split": "test",
            "env_name": "coinrun_proxy_fail",
            "model_file": "model.pth",
            "num_rollouts": 16,
            "level_seeds_file": "seeds.json",
            "video_logging_mode": "none",
            "wandb_mode": "disabled",
            "greedy": False,
        },
        "ood-stable",
        Path("/tmp/logs"),
    )

    assert "-en coinrun_proxy_fail" in command
    assert "--model_file model.pth" in command

import sys
from pathlib import Path

import torch

from YRC.algorithms.ood import OODAlgorithm
from lib.pyod.pyod.models.deep_svdd import DeepSVDD


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log_metrics(self, metrics, step=None):
        self.records.append((metrics, step))


def test_deep_svdd_logs_validation_loss_when_validation_data_is_provided():
    logger = RecordingLogger()
    clf = DeepSVDD(
        n_features=2,
        hidden_neurons=[4, 2],
        epochs=1,
        batch_size=4,
        feature_type="hidden",
        benchmark="procgen",
        input_shape=(1, 4),
        logger=logger,
    )

    train_data = torch.randn(8, 4)
    val_data = torch.randn(4, 4)

    clf.fit(X=train_data, X_threshold=train_data, X_val=val_data)

    assert len(logger.records) == 1
    metrics, step = logger.records[0]
    assert step == 1
    assert "train/loss" in metrics
    assert "val/loss" in metrics
    assert "val/best_loss" in metrics


def test_deep_svdd_omits_validation_loss_without_validation_data():
    logger = RecordingLogger()
    clf = DeepSVDD(
        n_features=2,
        hidden_neurons=[4, 2],
        epochs=1,
        batch_size=4,
        feature_type="hidden",
        benchmark="procgen",
        input_shape=(1, 4),
        logger=logger,
    )

    train_data = torch.randn(8, 4)

    clf.fit(X=train_data, X_threshold=train_data)

    metrics, _ = logger.records[0]
    assert "train/loss" in metrics
    assert "val/loss" not in metrics
    assert "val/best_loss" not in metrics


def test_run_svdd_train_command_passes_seed_file_and_validation_levels():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_svdd_train
    finally:
        sys.path.pop(0)

    command = run_svdd_train.build_sbatch_command(
        "job",
        {
            "wandb_group": "group",
            "config": "configs/procgen_ood.yaml",
            "name": "name",
            "env_name": "coinrun",
            "sim": "sim.pth",
            "weak": "weak.pth",
            "strong": "strong.pth",
            "cp_method": "DeepSVDD",
            "feature_type": "obs",
            "rollout_dir": "rollouts",
            "num_rollouts": 64,
            "level_seeds_file": "seeds/0.json",
            "svdd_val_levels": 64,
            "query_cost": 0,
            "seed": 6033,
            "rollout_max_levels": 128,
        },
    )

    assert "-level_seeds_file seeds/0.json" in command
    assert "-svdd_val_levels 64" in command
    assert "-rollout_max_levels 128" in command


def test_run_svdd_train_default_rollout_dir_matches_gather_output_layout():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_svdd_train
    finally:
        sys.path.pop(0)

    rollout_dir = run_svdd_train.get_rollout_dir(
        "coinrun", 0, "/rollouts", "rollouts-neurips"
    )

    assert rollout_dir == "/rollouts/rollouts-neurips/coinrun/gather_coinrun_exp0"


def test_run_gather_rollouts_exports_prefixed_rollout_output_dir():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_gather_rollouts
    finally:
        sys.path.pop(0)

    output_dir = run_gather_rollouts.get_rollout_output_dir(
        "/rollouts", "rollouts-neurips", "coinrun"
    )
    command = run_gather_rollouts.build_sbatch_command(
        "job",
        {
            "wandb_mode": "offline",
            "config": "configs/procgen_gather.yaml",
            "name": "gather_coinrun_exp0",
            "experiment_group": "rollouts-neurips_coinrun_exp0",
            "env_name": "coinrun",
            "random_percent": 0,
            "sim": "sim.pth",
            "weak": "weak.pth",
            "strong": "strong.pth",
            "use_bg": True,
            "seed": 6033,
            "level_seeds_file": "seeds/0.json",
            "query_cost": 0,
            "rollout_levels": None,
            "rollout_chunk_size": None,
            "output_dir": str(output_dir),
        },
    )

    assert output_dir == Path("/rollouts/rollouts-neurips/coinrun")
    assert 'export SM_OUTPUT_DIR="/rollouts/rollouts-neurips/coinrun"' in command


def test_ood_algorithm_stacks_rollouts_without_full_config_coord_policy():
    algorithm = OODAlgorithm(config=object(), env=None)
    rollout_obs = [torch.tensor([1.0]), torch.tensor([2.0])]

    stacked = algorithm._stack_rollout_obs(rollout_obs, "rollout_obs", "obs")

    assert torch.equal(stacked, torch.tensor([[1.0], [2.0]]))

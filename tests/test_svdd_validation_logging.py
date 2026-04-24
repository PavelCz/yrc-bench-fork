import sys
from pathlib import Path

import torch

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


def test_run_svdd_train_default_rollout_dir_matches_gather_output():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_svdd_train
    finally:
        sys.path.pop(0)

    rollout_dir = run_svdd_train.get_rollout_dir("coinrun", 0)

    assert rollout_dir == "gather_coinrun_exp0"

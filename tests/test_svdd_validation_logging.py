import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from YRC.algorithms.ood import OODAlgorithm
from lib.pyod.pyod.models.deep_svdd import DeepSVDD


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log_metrics(self, metrics, step=None):
        self.records.append((metrics, step))


def find_record_with_metric(logger, metric_name):
    for metrics, step in logger.records:
        if metric_name in metrics:
            return metrics, step
    raise AssertionError(f"Could not find logged metric {metric_name!r}")


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

    metrics, step = find_record_with_metric(logger, "train/loss")
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

    metrics, _ = find_record_with_metric(logger, "train/loss")
    assert "train/loss" in metrics
    assert "val/loss" not in metrics
    assert "val/best_loss" not in metrics


def test_deep_svdd_streaming_fit_sets_decision_scores_for_dataset_length():
    clf = DeepSVDD(
        n_features=2,
        hidden_neurons=[4, 2],
        epochs=1,
        batch_size=4,
        feature_type="hidden",
        benchmark="procgen",
        input_shape=(1, 4),
    )
    dataset = TensorDataset(torch.randn(9, 4))

    clf.fit(X=dataset, X_threshold=dataset)

    assert clf.decision_scores_.shape == (9,)


def test_deep_svdd_streaming_fit_logs_train_and_validation_metrics():
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
    dataset = TensorDataset(torch.randn(8, 4))
    val_data = torch.randn(4, 4)

    clf.fit(X=dataset, X_threshold=dataset, X_val=val_data)

    metrics, step = find_record_with_metric(logger, "train/loss")
    assert step == 1
    assert "train/loss" in metrics
    assert "train/best_loss" in metrics
    assert "val/loss" in metrics
    assert "val/best_loss" in metrics


def test_deep_svdd_logs_progress_before_epoch_summary():
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
    dataset = TensorDataset(torch.randn(8, 4))

    clf.fit(X=dataset, X_threshold=dataset)

    progress_record_index = next(
        index
        for index, (metrics, _) in enumerate(logger.records)
        if "train/batch_loss" in metrics
    )
    summary_record_index = next(
        index
        for index, (metrics, _) in enumerate(logger.records)
        if "train/loss" in metrics
    )

    assert progress_record_index < summary_record_index
    progress_metrics, _ = logger.records[progress_record_index]
    assert "train/running_loss" in progress_metrics
    assert "train/epoch_progress" in progress_metrics


class DummyWeakAgent:
    def __init__(self):
        self._device = torch.device("cpu")
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def get_hidden(self, obs):
        return torch.stack([obs.sum(dim=1), obs.mean(dim=1)], dim=1)


def test_ood_algorithm_hidden_streaming_batch_transform_uses_weak_agent():
    algorithm = OODAlgorithm(config=object(), env=None)
    weak_agent = DummyWeakAgent()
    policy = SimpleNamespace(device=torch.device("cpu"))
    envs = {"train": SimpleNamespace(weak_agent=weak_agent)}

    transform = algorithm._make_streaming_batch_transform(policy, envs, "hidden")
    hidden = transform(torch.tensor([[1.0, 3.0], [2.0, 4.0]]))

    assert weak_agent.eval_called
    assert hidden.shape == (2, 2)
    assert torch.equal(hidden, torch.tensor([[4.0, 2.0], [6.0, 3.0]]))


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
            "output_dir": "/svdd/prefix",
        },
        Path("/logs/svdd_train/prefix/2026-04-29"),
    )

    assert "-level_seeds_file seeds/0.json" in command
    assert "-svdd_val_levels 64" in command
    assert "-rollout_max_levels 128" in command
    assert 'export SM_OUTPUT_DIR="/svdd/prefix"' in command
    assert "#SBATCH --output=/logs/svdd_train/prefix/2026-04-29/%x_%j.out" in command


def test_run_svdd_train_default_rollout_dir_matches_gather_output_layout():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_svdd_train
    finally:
        sys.path.pop(0)

    rollout_dir = run_svdd_train.get_rollout_dir(
        "coinrun", 0, "/rollouts", run_svdd_train.TRAIN_DEFAULTS["rollouts_prefix"]
    )

    assert run_svdd_train.TRAIN_DEFAULTS["rollouts_prefix"] == "neurips03"
    assert run_svdd_train.TRAIN_DEFAULTS["rollout_max_levels"] == 1024
    assert rollout_dir == "/rollouts/neurips03/coinrun/gather_coinrun_exp0"


def test_run_svdd_train_expected_model_path_includes_prefix():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_svdd_train
    finally:
        sys.path.pop(0)

    output_dir = run_svdd_train.get_svdd_output_dir("/trained_svdd", "neurips02")
    model_file = run_svdd_train.get_expected_model_file(
        output_dir, "svdd_coinrun_image_exp0"
    )

    assert output_dir == Path("/trained_svdd/neurips02")
    assert model_file == Path(
        "/trained_svdd/neurips02/svdd_coinrun_image_exp0/trained.joblib"
    )


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
    assert "-rollout_chunk_size" not in command


def test_run_gather_rollouts_defaults_to_neurips_extra_seed_dir():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_gather_rollouts
    finally:
        sys.path.pop(0)

    seeds_dir = run_gather_rollouts.get_default_level_seeds_dir(
        "/nas/ucb/czempin/data/goal-misgen/seeds/icml"
    )

    assert seeds_dir == Path(
        "/nas/ucb/czempin/data/goal-misgen/seeds/neurips_extra_ood_train_1024"
    )
    assert run_gather_rollouts.GATHER_DEFAULTS["rollout_chunk_size"] is None


def test_run_gather_rollouts_allows_level_seed_file_override():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_gather_rollouts
    finally:
        sys.path.pop(0)

    level_seeds_file = run_gather_rollouts.get_level_seeds_file(
        0, "/canonical/seeds", Path("/extra/seeds/0.json")
    )

    assert level_seeds_file == Path("/extra/seeds/0.json")


def test_ood_algorithm_stacks_rollouts_without_full_config_coord_policy():
    algorithm = OODAlgorithm(config=object(), env=None)
    rollout_obs = [torch.tensor([1.0]), torch.tensor([2.0])]

    stacked = algorithm._stack_rollout_obs(rollout_obs, "rollout_obs", "obs")

    assert torch.equal(stacked, torch.tensor([[1.0], [2.0]]))

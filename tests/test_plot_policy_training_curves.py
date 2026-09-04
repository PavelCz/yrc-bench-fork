import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from analyzing.plot_policy_training_curves import (
    _active_panels,
    aggregate_metric,
    load_training_csv,
    plot_training_curves,
)


def test_aggregate_metric_averages_aligned_timesteps():
    dfs = [
        pd.DataFrame(
            {
                "timesteps": [10, 20, 30],
                "mean_episode_rewards": [1.0, 2.0, 3.0],
            }
        ),
        pd.DataFrame(
            {
                "timesteps": [10, 20, 30],
                "mean_episode_rewards": [3.0, 4.0, 5.0],
            }
        ),
    ]

    stats = aggregate_metric(dfs, "mean_episode_rewards")

    assert list(stats.index) == [10, 20, 30]
    assert list(stats["mean"]) == [2.0, 3.0, 4.0]
    assert list(stats["q25"]) == [1.5, 2.5, 3.5]
    assert list(stats["q75"]) == [2.5, 3.5, 4.5]


def test_load_training_csv_reads_required_columns(tmp_path):
    csv_path = tmp_path / "log-append.csv"
    pd.DataFrame(
        {
            "timesteps": [1],
            "mean_episode_rewards": [0.5],
            "val_mean_episode_rewards": [0.25],
        }
    ).to_csv(csv_path, index=False)

    df = load_training_csv(csv_path)

    assert list(df["timesteps"]) == [1]
    assert list(df["mean_episode_rewards"]) == [0.5]
    assert list(df["val_mean_episode_rewards"]) == [0.25]


def test_active_panels_include_aux_metrics_when_present():
    dfs = [
        pd.DataFrame(
            {
                "timesteps": [1, 2],
                "mean_episode_rewards": [1.0, 2.0],
                "val_mean_episode_rewards": [0.5, 1.5],
                "val_id_mean_episode_rewards": [0.4, 1.4],
                "val_ood_mean_episode_rewards": [0.6, 1.6],
                "val_mean_oracle_regret": [0.2, 0.1],
                "val_id_mean_oracle_regret": [0.3, 0.2],
                "val_ood_mean_oracle_regret": [0.1, 0.0],
            }
        )
    ]

    ylabels = [ylabel for ylabel, _ in _active_panels(dfs)]

    assert ylabels == [
        "Mean episode return",
        "Val mean episode return",
        "Oracle regret",
    ]


def test_plot_training_curves_uses_three_subplots_for_aux_metrics(tmp_path):
    csv_path = tmp_path / "log-append.csv"
    pd.DataFrame(
        {
            "timesteps": [1, 2],
            "mean_episode_rewards": [1.0, 2.0],
            "val_mean_episode_rewards": [0.5, 1.5],
            "val_id_mean_episode_rewards": [0.4, 1.4],
            "val_ood_mean_episode_rewards": [0.6, 1.6],
            "val_mean_oracle_regret": [0.2, 0.1],
            "val_id_mean_oracle_regret": [0.3, 0.2],
            "val_ood_mean_oracle_regret": [0.1, 0.0],
        }
    ).to_csv(csv_path, index=False)

    fig, axes = plot_training_curves(
        [csv_path],
        output_path=str(tmp_path / "curves.png"),
        show_plot=False,
    )

    assert fig is not None
    assert len(axes) == 3
    assert (tmp_path / "curves.png").exists()
    plt.close(fig)

import pandas as pd

from analyzing.plot_policy_training_curves import aggregate_metric, load_training_csv


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

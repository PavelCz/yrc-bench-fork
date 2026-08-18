import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def _smooth_series(values: pd.Series, window: int) -> pd.Series:
    """Return a centered rolling mean, or the original series if window <= 1."""
    if window <= 1 or len(values) < window:
        return values
    return values.rolling(window=window, min_periods=1, center=True).mean()


def _plot_reward_curve(
    ax,
    timesteps,
    rewards,
    *,
    color: str,
    label: str,
    marker: str,
    smooth_window: int,
):
    rewards = rewards.astype(float)
    if smooth_window > 1:
        ax.plot(
            timesteps,
            rewards,
            color=color,
            linewidth=1,
            alpha=0.25,
            marker=marker,
            markersize=3,
        )
        ax.plot(
            timesteps,
            _smooth_series(rewards, smooth_window),
            label=label,
            color=color,
            linewidth=2.5,
        )
    else:
        ax.plot(
            timesteps,
            rewards,
            label=label,
            linewidth=2,
            marker=marker,
            markersize=4,
            alpha=0.8,
            color=color,
        )


def plot_training_curves(
    csv_path,
    output_path="training_curves.png",
    show_plot=True,
    smooth_window: int = 0,
):
    """
    Plot training and validation mean episode rewards over timesteps.

    Args:
        csv_path: Path to the CSV file containing training logs
        output_path: Path to save the output plot
        show_plot: Whether to display the plot interactively
        smooth_window: Rolling-average window in log points (0 or 1 = no smoothing)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

    # Check if required columns exist
    required_cols = ["timesteps", "mean_episode_rewards", "val_mean_episode_rewards"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None, None

    # Set up the plot style
    plt.style.use("default")  # Use default style for better compatibility
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter out NaNs for training rewards and plot
    train_mask = df["mean_episode_rewards"].notna()
    if train_mask.any():
        train_timesteps = df.loc[train_mask, "timesteps"]
        train_rewards = df.loc[train_mask, "mean_episode_rewards"]
        _plot_reward_curve(
            ax,
            train_timesteps,
            train_rewards,
            color="blue",
            label="Training Mean Episode Rewards",
            marker="o",
            smooth_window=smooth_window,
        )
        print(f"Training curve: {len(train_rewards)} valid points")
    else:
        print("No valid training data points found")

    # Filter out NaNs for validation rewards and plot
    val_mask = df["val_mean_episode_rewards"].notna()
    if val_mask.any():
        val_timesteps = df.loc[val_mask, "timesteps"]
        val_rewards = df.loc[val_mask, "val_mean_episode_rewards"]
        _plot_reward_curve(
            ax,
            val_timesteps,
            val_rewards,
            color="red",
            label="Validation Mean Episode Rewards",
            marker="s",
            smooth_window=smooth_window,
        )
        print(f"Validation curve: {len(val_rewards)} valid points")
    else:
        print("No valid validation data points found")

    # Check if we have any data to plot
    if not train_mask.any() and not val_mask.any():
        print("No valid data points found for either curve")
        return None, None

    # Customize the plot
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Mean Episode Rewards", fontsize=14)
    ax.set_title("Training and Validation Mean Episode Rewards Over Time", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis to show timesteps in a readable way
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add some styling
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved as: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Show the plot if requested
    if show_plot:
        plt.show()

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from CSV log files"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/pavel/data/goal-misgen/policy/train/coinrun/coinrun_hard_bg/2025-08-22__23-20-39__seed_6033/log-append.csv",
        help="Path to the CSV file containing training logs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_curves.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the plot interactively"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Rolling-average window in log points (default: 0, no smoothing)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.csv_path).exists():
        print(f"CSV file not found: {args.csv_path}")
        return

    # Create the plot
    fig, ax = plot_training_curves(
        args.csv_path,
        args.output,
        not args.no_show,
        smooth_window=args.smooth,
    )

    if fig is None:
        print("Failed to create plot")
        return


if __name__ == "__main__":
    main()

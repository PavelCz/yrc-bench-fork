from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_afhp(name_order=None):
    """
    Plot AFHP (Ask for Help Percentage) vs performance.

    Args:
        name_order: List of method names to plot in order. If None, uses all available names.
    """
    results = extract_results()

    # Collect first and last performance values for all curves
    first_performances = []
    last_performances = []

    name_map = {
        "latent-svdd": "Latent SVDD",
        "random": "Random",
        "patient-ae": "Autoencoder",
        "center-focused": "Center-focused AE",
        "deep-svdd": "DeepSVDD",
    }

    # If name_order is None, use all available names
    if name_order is None:
        name_order = list(results.keys())

    # Clear previous plot
    plt.clf()

    for name in name_order:
        if name not in results:
            print(f"Warning: {name} not found in evals, skipping...")
            continue

        data_path = results[name]

        eval_data = np.load(data_path, allow_pickle=True)
        x, y = extract_x_and_y_values(eval_data)

        # desired_percentiles = eval_data["desired_percentiles"]

        # Store first and last performance values
        first_performances.append(y[0])
        last_performances.append(y[-1])

        if name in name_map:
            label = name_map[name]
        else:
            label = name

        sns.lineplot(x=x, y=y, label=label, marker="o")

    # Calculate means
    mean_first_performance = np.mean(first_performances)
    mean_last_performance = np.mean(last_performances)

    # Add horizontal lines
    plt.axhline(
        y=mean_first_performance,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Weak Agent",
    )
    plt.axhline(
        y=mean_last_performance,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Oracle",
    )

    plt.xlabel("Ask for help percentage")
    plt.ylabel("Mean return")
    plt.title("Mean return vs. ask for help percentage")
    plt.legend()
    # plt.savefig("afhp_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def extract_results():
    base_path = Path("/home/pavel/data/goal-misgen/tmp")
    eval_path = base_path / "27-easy-policy"

    prefix_filter = "24-easy-policy"

    evals = {}

    for child in eval_path.iterdir():
        if child.is_dir():
            method_name = child.name
            if (child / "eval_runs").exists():
                for grandchild in (child / "eval_runs").iterdir():
                    for grandgrandchild in grandchild.iterdir():
                        if (
                            grandgrandchild.is_file()
                            and grandgrandchild.suffix == ".npz"
                            and grandchild.stem.startswith(f"eval-{prefix_filter}")
                        ):
                            evals[method_name] = grandgrandchild
    return evals


def extract_x_and_y_values(data):
    x = data["afhps"]
    y = data["performances"]
    return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Plot AFHP (Ask for Help Percentage) vs performance"
    )
    parser.add_argument(
        "--name_order",
        "-n",
        type=str,
        default=None,
        help="Comma-separated list of method names to plot in order. If not specified, uses all available names.",
    )

    args = parser.parse_args()

    # Parse name_order if provided
    name_order = None
    if args.name_order:
        name_order = [name.strip() for name in args.name_order.split(",")]
        print(f"Using specified order: {name_order}")
    else:
        print("Using all available method names")

    plot_afhp(name_order)


if __name__ == "__main__":
    main()

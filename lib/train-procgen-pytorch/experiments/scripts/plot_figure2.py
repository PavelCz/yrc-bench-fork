import argparse
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def listdir(path):
    return [os.path.join(path, d) for d in os.listdir(path)]


def path_to_rand_percent(path: str) -> int:
    """extract integer rand_percent from path info
    path must end with integer rand_percent"""
    out = path[-3:]
    while not out.isdigit():
        out = out[1:]
    return int(out)


def seed_collect_freq(seed: int, van_df: pd.DataFrame) -> float:
    """returns frequency at which agent collects the inv coin
    in the vanilla environment, for env seed"""
    idx = van_df['seed'] == seed
    return float(np.mean(van_df.loc[idx]["inv_coin_collected"]))


def get_good_seed_df(df: pd.DataFrame, good_seeds: np.ndarray) -> pd.DataFrame:
    """
    given a dataframe with column 'seed', return new dataframe that is a subset
    of the old one, and includes only rows with good seeds.
    """
    good_seed_idx = [seed in good_seeds for seed in df['seed']]
    return df.loc[good_seed_idx]


def main(args):
    sns.set()

    # Base directory where the output of get-fig2-data.sh is stored
    results_dir = Path("experiments/results/").resolve()

    # Use the frequency of collecting the invisible coin in the 'vanilla' setting to
    # decide which levels to filter out.
    vanilla_coinrun_resdir = results_dir / "test_rand_percent_0"
    van_df = extract_metrics(vanilla_coinrun_resdir)
    max_collect_freq = 0.1

    collect_freqs = list(map(lambda s: seed_collect_freq(s, van_df), range(np.max(van_df['seed']))))
    collect_freqs = np.array(collect_freqs)
    (good_seeds,) = np.nonzero(collect_freqs < max_collect_freq)

    rand_percents = [path_to_rand_percent(file.name) for file in results_dir.iterdir() if file.name.startswith("test_rand")]
    rand_percents.sort()
    joint_rp_paths = [results_dir / f"test_rand_percent_{rp}" for rp in rand_percents[:-1]]

    # sweep over training rand_percent
    # TODO: Re-implement iterating over multiple training rand percentages.
    # To do this, look into the old version of this script from the git history.
    test_rp100_resdir = results_dir / "test_rand_percent_100"
    dataframes = {
        path_to_rand_percent(test_rp100_resdir.name): extract_metrics(test_rp100_resdir)
    }
    dataframes = {k: get_good_seed_df(df, good_seeds) for k, df in dataframes.items()}
    reach_end_freqs = {k: np.mean(df["inv_coin_collected"]) for k, df in dataframes.items()}

    data = list(reach_end_freqs.items())
    data.sort()
    data = np.array(data)

    # sweep over training & test rand_percent jointly
    # measure how often model dies or times out, ie not gets coin
    dataframes = {
        path_to_rand_percent(path.name): extract_metrics(path) for path in joint_rp_paths
    }
    dataframes = {k: get_good_seed_df(df, good_seeds) for k, df in dataframes.items()}
    fail_to_get_coin_freq = {k: 1 - np.mean(df["coin_collected"]) for k, df in dataframes.items()}

    joint_data = list(fail_to_get_coin_freq.items())
    joint_data.sort()
    joint_data = np.array(joint_data)

    baseline_vanilla_df = get_good_seed_df(van_df, good_seeds)
    prob_of_reaching_end_without_inv_coin = np.mean(baseline_vanilla_df["coin_collected"] == 1)

    figpath = "./"

    fig, ax = plt.subplots(figsize=[6, 2.5])
    plt.axhline(y=prob_of_reaching_end_without_inv_coin * 100, linestyle="--", color="tab:grey", label="Maximum possible OR frequency")

    x, y = joint_data.T
    ax.plot(x, y*100, "--o", label="IID Robustness Failure", color="tab:orange")

    x, y = data.T
    ax.plot(x, y*100, "--o", label="Objective Robustness Failure", color="tab:blue")

    # plt.ylim(50, 101)
    plt.ylabel("Frequency (%)")
    plt.xlabel("Probability (%) of a level with randomized coin.")
    plt.legend()

    plt.savefig(figpath + "coinrun_freq.pdf")

def extract_metrics(dir: Path) -> pd.DataFrame:
    """
    Extract metrics from the vanilla coinrun results directory.
    """
    subfolders = [
        f for f in dir.iterdir() if f.is_dir()
        ]
    if len(subfolders) == 0 or len(subfolders) > 1:
        raise ValueError(f"Expected 1 subfolder in {dir}, got {len(subfolders)}")
    sub_dir = dir / subfolders[0]
    
    csv_files = [f for f in sub_dir.iterdir() if f.suffix == '.csv']
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .csv file found in {sub_dir}")
    elif len(csv_files) > 1:
        raise ValueError(
            f"Expected 1 .csv file in {dir}, found {len(csv_files)}: "
            f"{csv_files}"
        )
    df = pd.read_csv(csv_files[0])
    return df


if __name__ == "__main__":
    # CLI: allow overriding the vanilla results dir and the fixed 100% test dir (relative to results_dir)
    parser = argparse.ArgumentParser(description="Plot Figure 2 from collected evaluation CSVs")
    parser.add_argument(
        "--vanilla_resdir",
        type=str,
        default="test_rand_percent_0/train_rand_percent_0/",
        help="Path RELATIVE to results_dir containing vanilla coinrun results (metrics.csv)"
    )
    parser.add_argument(
        "--test_rp100_resdir",
        type=str,
        default="test_rand_percent_100",
        help="Path RELATIVE to results_dir containing test results at 100% random percent"
    )
    args = parser.parse_args()
    main(args)

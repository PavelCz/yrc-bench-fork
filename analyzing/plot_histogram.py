#!/usr/bin/env python3
"""
Interactive script to plot histograms of episode-level metrics for specific evaluation checkpoints.
"""

# TODO Remove after this program no longer supports Python 3.8.*
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from analyzing.utils import extract_results


def get_episode_level_data(element: dict, key: str) -> np.ndarray:
    """
    Extract episode-level data for a given key from an evaluation checkpoint.
    
    Args:
        element: A single element from data["meta"] representing one checkpoint
        key: The metric key to extract
        
    Returns:
        Array of values, one per episode
    """
    test_summary = element["summary"]["test"]
    
    if key == "episode_length":
        return np.array(test_summary["episode_lengths"])
    elif key == "raw_return":
        return np.array(test_summary["raw_returns"])
    elif key == "level_ood_gt":
        return np.array(test_summary["level_ood_gt"])
    elif key == "level_ood_pred":
        return np.array(test_summary["level_ood_pred"])
    elif key == "first_ood_timestep":
        # Filter out None values for timesteps where OOD was never predicted
        timesteps = test_summary["first_ood_timestep"]
        valid_timesteps = [t for t in timesteps if t is not None]
        return np.array(valid_timesteps)
    elif key == "ood_prediction_correctness":
        # Whether the OOD prediction matches ground truth (per episode)
        preds = np.array(test_summary["level_ood_pred"])
        gts = np.array(test_summary["level_ood_gt"])
        return (preds == gts).astype(int)
    else:
        raise ValueError(f"Unknown key: {key}")


def interactive_histogram_plotter():
    """Main function for interactive histogram plotting."""
    parser = argparse.ArgumentParser(
        description="Interactive histogram plotter for episode-level metrics"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing the evaluation files.",
    )
    parser.add_argument(
        "--prefix_filter",
        default=None,
        type=str,
        help="Prefix filter for the evaluation files.",
    )
    parser.add_argument(
        "--key",
        type=str,
        required=True,
        help=(
            "Metric key to plot. Options: episode_length, raw_return, level_ood_gt, "
            "level_ood_pred, first_ood_timestep, ood_prediction_correctness"
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the histogram (default: 30)",
    )
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    # Step 1: Extract all runs
    print("Extracting available runs...")
    results = extract_results(eval_dir, args.prefix_filter)
    
    if not results:
        print("No runs found!")
        return
    
    # Step 2: Display runs and let user select
    print("\nAvailable runs:")
    run_names = list(results.keys())
    for idx, name in enumerate(run_names):
        print(f"  [{idx}] {name}")
    
    while True:
        try:
            selection = input(f"\nSelect a run (0-{len(run_names)-1}): ")
            run_idx = int(selection)
            if 0 <= run_idx < len(run_names):
                break
            else:
                print(f"Please enter a number between 0 and {len(run_names)-1}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_run = run_names[run_idx]
    data_path = results[selected_run]
    
    print(f"\nLoading data from: {data_path}")
    
    # Step 3: Load the data
    eval_data = np.load(data_path, allow_pickle=True)
    
    # Step 4: Display ood_pred_percentage for each checkpoint
    print("\nCheckpoints with OOD prediction percentages:")
    ood_percentages = []
    
    for idx, element in enumerate(eval_data["meta"]):
        level_ood_pred = element["summary"]["test"]["level_ood_pred"]
        percentage = sum(level_ood_pred) / len(level_ood_pred) * 100
        ood_percentages.append(percentage)
        
        # Also show AFHP and performance if available
        afhp = eval_data["afhps"][idx] if idx < len(eval_data["afhps"]) else "N/A"
        perf = eval_data["performances"][idx] if idx < len(eval_data["performances"]) else "N/A"
        
        print(f"  [{idx}] OOD%: {percentage:.2f}%, AFHP: {afhp}, Performance: {perf}")
    
    # Step 5: Let user select a checkpoint
    while True:
        try:
            selection = input(f"\nSelect a checkpoint (0-{len(ood_percentages)-1}): ")
            checkpoint_idx = int(selection)
            if 0 <= checkpoint_idx < len(ood_percentages):
                break
            else:
                print(f"Please enter a number between 0 and {len(ood_percentages)-1}")
        except ValueError:
            print("Please enter a valid number")
    
    # Step 6: Extract data for the selected checkpoint and key
    selected_element = eval_data["meta"][checkpoint_idx]
    
    try:
        values = get_episode_level_data(selected_element, args.key)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if len(values) == 0:
        print(f"No data available for key '{args.key}' at checkpoint {checkpoint_idx}")
        return
    
    # Step 7: Plot histogram
    print(f"\nPlotting histogram for '{args.key}' at checkpoint {checkpoint_idx}")
    print(f"  OOD prediction percentage: {ood_percentages[checkpoint_idx]:.2f}%")
    print(f"  Number of episodes: {len(values)}")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Std: {np.std(values):.2f}")
    print(f"  Min: {np.min(values):.2f}")
    print(f"  Max: {np.max(values):.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=args.bins, edgecolor='black', alpha=0.7)
    plt.xlabel(args.key.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title(
        f'{args.key.replace("_", " ").title()} Distribution\n'
        f'Run: {selected_run}, Checkpoint: {checkpoint_idx}, '
        f'OOD%: {ood_percentages[checkpoint_idx]:.1f}%'
    )
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    interactive_histogram_plotter()


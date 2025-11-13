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


def plot_single_run(run_names: list, results: dict, key: str, bins: int, success_only: bool = False):
    """Plot histogram for a single selected run and checkpoint."""
    # Step 2: Display runs and let user select
    print("\nAvailable runs:")
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
        values = get_episode_level_data(selected_element, key, success_only)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if len(values) == 0:
        filter_msg = " (success only)" if success_only else ""
        print(f"No data available for key '{key}' at checkpoint {checkpoint_idx}{filter_msg}")
        return
    
    # Step 7: Plot histogram
    filter_msg = " (success only)" if success_only else ""
    print(f"\nPlotting histogram for '{key}' at checkpoint {checkpoint_idx}{filter_msg}")
    print(f"  OOD prediction percentage: {ood_percentages[checkpoint_idx]:.2f}%")
    print(f"  Number of episodes: {len(values)}")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Std: {np.std(values):.2f}")
    print(f"  Min: {np.min(values):.2f}")
    print(f"  Max: {np.max(values):.2f}")
    
    plt.figure(figsize=(10, 6))
    counts, bin_edges, patches = plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    
    # Set x-axis ticks to show bin edges
    plt.xticks(bin_edges, rotation=45, ha='right')
    
    plt.xlabel(key.replace('_', ' ').title())
    plt.ylabel('Frequency')
    title_suffix = " (Success Only)" if success_only else ""
    plt.title(
        f'{key.replace("_", " ").title()} Distribution{title_suffix}\n'
        f'Run: {selected_run}, Checkpoint: {checkpoint_idx}, '
        f'OOD%: {ood_percentages[checkpoint_idx]:.1f}%'
    )
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_compare_runs(run_names: list, results: dict, key: str, bins: int, success_only: bool = False):
    """Plot histograms for selected checkpoints from multiple runs."""
    print("\n=== Multi-Run Comparison Mode ===")
    filter_msg = " (success only)" if success_only else ""
    print(f"You will select a checkpoint for each run to compare{filter_msg}.\n")
    
    # Collect data for each run
    all_data = []
    all_labels = []
    
    for run_name in run_names:
        data_path = results[run_name]
        print(f"\n--- Run: {run_name} ---")
        print(f"Loading data from: {data_path}")
        
        # Load the data
        eval_data = np.load(data_path, allow_pickle=True)
        
        # Display ood_pred_percentage for each checkpoint
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
        
        # Let user select a checkpoint
        while True:
            try:
                selection = input(f"\nSelect a checkpoint for '{run_name}' (0-{len(ood_percentages)-1}): ")
                checkpoint_idx = int(selection)
                if 0 <= checkpoint_idx < len(ood_percentages):
                    break
                else:
                    print(f"Please enter a number between 0 and {len(ood_percentages)-1}")
            except ValueError:
                print("Please enter a valid number")
        
        # Extract data for the selected checkpoint
        selected_element = eval_data["meta"][checkpoint_idx]
        
        try:
            values = get_episode_level_data(selected_element, key, success_only)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        
        if len(values) == 0:
            filter_msg = " (success only)" if success_only else ""
            print(f"No data available for key '{key}' at checkpoint {checkpoint_idx}{filter_msg}")
            continue
        
        # Store data and label
        all_data.append(values)
        label = f"{run_name} (OOD: {ood_percentages[checkpoint_idx]:.1f}%)"
        all_labels.append(label)
        
        print(f"  Selected checkpoint {checkpoint_idx}")
        print(f"  Number of episodes: {len(values)}")
        print(f"  Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")
    
    if len(all_data) == 0:
        print("\nNo valid data collected. Exiting.")
        return
    
    # Calculate common bins based on all data
    all_values_combined = np.concatenate(all_data)
    min_val = np.min(all_values_combined)
    max_val = np.max(all_values_combined)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Plot all histograms
    print(f"\nPlotting {len(all_data)} histograms...")
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    
    for idx, (values, label) in enumerate(zip(all_data, all_labels)):
        plt.hist(values, bins=bin_edges, alpha=0.5, label=label, 
                edgecolor='black', linewidth=0.5, color=colors[idx])
    
    # Set x-axis ticks to show bin edges
    plt.xticks(bin_edges, rotation=45, ha='right')
    
    plt.xlabel(key.replace('_', ' ').title())
    plt.ylabel('Frequency')
    title_suffix = " (Success Only)" if success_only else ""
    plt.title(f'{key.replace("_", " ").title()} Distribution Comparison{title_suffix}')
    plt.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_episode_level_data(element: dict, key: str, success_only: bool = False) -> np.ndarray:
    """
    Extract episode-level data for a given key from an evaluation checkpoint.
    
    Args:
        element: A single element from data["meta"] representing one checkpoint
        key: The metric key to extract
        success_only: If True, only return values for episodes with reward > 0
        
    Returns:
        Array of values, one per episode (filtered if success_only=True)
    """
    test_summary = element["summary"]["test"]
    
    # Get the requested data
    if key == "episode_length":
        values = np.array(test_summary["episode_lengths"])
    elif key == "raw_return":
        values = np.array(test_summary["raw_returns"])
    elif key == "level_ood_gt":
        values = np.array(test_summary["level_ood_gt"])
    elif key == "level_ood_pred":
        values = np.array(test_summary["level_ood_pred"])
    elif key == "first_ood_timestep":
        # Filter out None values for timesteps where OOD was never predicted
        timesteps = test_summary["first_ood_timestep"]
        valid_timesteps = [t for t in timesteps if t is not None]
        values = np.array(valid_timesteps)
    elif key == "ood_prediction_correctness":
        # Whether the OOD prediction matches ground truth (per episode)
        preds = np.array(test_summary["level_ood_pred"])
        gts = np.array(test_summary["level_ood_gt"])
        values = (preds == gts).astype(int)
    else:
        raise ValueError(f"Unknown key: {key}")
    
    # Filter by success if requested
    if success_only and key != "first_ood_timestep":
        # For first_ood_timestep, we already filtered out None values, 
        # and we can't easily map back to returns
        raw_returns = np.array(test_summary["raw_returns"])
        success_mask = raw_returns > 0
        values = values[success_mask]
    elif success_only and key == "first_ood_timestep":
        # For first_ood_timestep, we need to handle differently
        # Get returns and first_ood_timestep in parallel
        raw_returns = np.array(test_summary["raw_returns"])
        timesteps = test_summary["first_ood_timestep"]
        # Only keep timesteps where return > 0 and timestep is not None
        valid_timesteps = [t for t, r in zip(timesteps, raw_returns) if t is not None and r > 0]
        values = np.array(valid_timesteps)
    
    return values


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
    parser.add_argument(
        "--compare_runs",
        action="store_true",
        help="Compare multiple runs by plotting histograms for selected checkpoints from each run",
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Only include episodes where the reward was greater than 0",
    )
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    # Step 1: Extract all runs
    print("Extracting available runs...")
    results = extract_results(eval_dir, args.prefix_filter)
    
    if not results:
        print("No runs found!")
        return
    
    run_names = list(results.keys())
    
    if args.compare_runs:
        # Multi-run comparison mode
        plot_compare_runs(run_names, results, args.key, args.bins, args.success_only)
    else:
        # Single run mode (original behavior)
        plot_single_run(run_names, results, args.key, args.bins, args.success_only)


if __name__ == "__main__":
    interactive_histogram_plotter()


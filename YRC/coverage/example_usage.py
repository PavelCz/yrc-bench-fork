"""
Example usage of the binary search coverage sampler.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

from YRC.coverage.binary_search import BinarySearchSampler


def example_monotonic_function(x: float) -> float:
    """
    Example monotonic function simulating threshold to AFHP mapping.

    This simulates a sigmoid-like relationship where:
    - Low percentiles (high thresholds) → Low AFHP
    - High percentiles (low thresholds) → High AFHP
    """
    # Add some non-linearity to make it interesting
    # Transform x from [0, 100] to [-6, 6] for sigmoid
    z = (x - 50) / 8
    sigmoid = 1 / (1 + np.exp(-z))

    # Scale to [0, 100] with some noise
    afhp = sigmoid * 95 + 2.5  # Maps roughly to [2.5, 97.5]

    # Add small amount of noise
    noise = np.random.normal(0, 1)
    return np.clip(afhp + noise, 0, 100)


def example_evaluation_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
    """
    Example evaluation function that simulates policy evaluation.

    Returns AFHP and metadata including simulated return.
    """
    # For this example, threshold is actually the percentile
    afhp = example_monotonic_function(threshold)

    # Simulate return as a function of AFHP with extremely steep initial rise and quick plateau
    # More help → better return, but with very diminishing returns
    base_return = 25  # Weak agent alone
    max_return = 90   # Strong agent alone
    
    # Create an extremely steep logarithmic curve that plateaus very quickly
    if afhp <= 0:
        return_value = base_return
    else:
        # Scale AFHP to make the curve much steeper at the beginning
        scaled_afhp = afhp / 100.0  # Normalize to [0, 1]
        
        # Use an even steeper logarithmic function
        # Most of the benefit should come from the first 10-15% AFHP
        k = 100  # Much higher k for extremely steep initial rise
        log_factor = np.log(1 + k * scaled_afhp) / np.log(1 + k)
        
        # Apply additional steepening transformation
        # Use a power function to make it even more concentrated at the beginning
        steepness_power = 0.3  # Lower power = more concentrated benefit early
        transformed_factor = log_factor ** steepness_power
        
        return_value = base_return + (max_return - base_return) * transformed_factor
    
    # Add some noise
    return_value += np.random.normal(0, 1.5)
    
    # Ensure return stays within reasonable bounds
    return_value = np.clip(return_value, base_return - 5, max_return + 5)

    metadata = {
        "return_mean": return_value,
        "return_std": np.random.uniform(1, 3),
        "threshold_used": threshold,
    }

    return afhp, metadata

def compare_with_uniform_sampling():
    """Compare binary search with uniform sampling."""

    print("\nComparing Binary Search vs Uniform Sampling")
    print("=" * 50)

    num_bins = 10
    num_evals = 12  # Slightly more than num_bins

    # Run enhanced binary search sampler with return refinement
    bs_sampler = BinarySearchSampler(
        eval_function=example_evaluation_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        return_bins=8,  # Enable return gap filling
        max_additional_evals=10,
    )
    bs_samples = bs_sampler.run_with_return_refinement()
    bs_summary = bs_sampler.get_coverage_summary()

    # Run uniform sampling for comparison
    uniform_x = np.linspace(0, 100, num_evals)
    uniform_samples = []
    uniform_sample_data = []  # For visualization (afhp, return)
    for x in uniform_x:
        afhp, metadata = example_evaluation_function(x)
        uniform_samples.append((x, afhp))
        uniform_sample_data.append((afhp, metadata["return_mean"]))

    # Compute uniform coverage
    uniform_bins_filled = set()
    for _, afhp in uniform_samples:
        bin_idx = int(afhp // 10) if afhp < 100 else 9
        uniform_bins_filled.add(bin_idx)

    uniform_coverage = 100.0 * len(uniform_bins_filled) / num_bins

    print("\nEnhanced Binary Search:")
    print(f"  Coverage: {bs_summary['coverage_percentage']:.1f}%")
    print(f"  Evaluations: {bs_summary['total_evaluations']}")
    print(f"  Return refinement samples: {len(bs_sampler.get_return_refinement_samples())}")

    print("\nUniform Sampling:")
    print(f"  Coverage: {uniform_coverage:.1f}%")
    print(f"  Evaluations: {num_evals}")

    # Create a single figure with all comparison plots
    plt.figure(figsize=(20, 12))

    # Plot 1: Coverage comparison
    ax1 = plt.subplot(2, 3, 1)

    # Filter out None samples
    valid_samples = [s for s in bs_samples if s is not None]

    # Binary search data
    bs_output_values = [s.output_value for s in valid_samples]
    bs_returns = [s.metadata.get("return_mean", 0) for s in valid_samples]

    # Sort by output value for line plot
    sorted_indices = np.argsort(bs_output_values)
    sorted_outputs = [bs_output_values[i] for i in sorted_indices]
    sorted_returns = [bs_returns[i] for i in sorted_indices]

    # Plot binary search
    ax1.plot(
        sorted_outputs, sorted_returns, "b-", linewidth=2, label="Binary search curve"
    )
    ax1.scatter(
        bs_output_values,
        bs_returns,
        c="red",
        s=50,
        zorder=5,
        label="Binary search points",
    )

    # Plot uniform samples
    uniform_afhps = [afhp for afhp, _ in uniform_sample_data]
    uniform_returns = [ret for _, ret in uniform_sample_data]

    # Sort for line plot
    sorted_uniform_indices = np.argsort(uniform_afhps)
    sorted_uniform_afhps = [uniform_afhps[i] for i in sorted_uniform_indices]
    sorted_uniform_returns = [uniform_returns[i] for i in sorted_uniform_indices]

    ax1.plot(
        sorted_uniform_afhps,
        sorted_uniform_returns,
        "g--",
        linewidth=2,
        alpha=0.7,
        label="Uniform curve",
    )
    ax1.scatter(
        uniform_afhps,
        uniform_returns,
        c="green",
        s=50,
        zorder=4,
        alpha=0.7,
        label="Uniform points",
    )

    # Plot return refinement samples
    refinement_samples = bs_sampler.get_return_refinement_samples()
    if refinement_samples:
        refinement_afhps = [s.output_value for s in refinement_samples]
        refinement_returns = [s.metadata.get("return_mean", 0) for s in refinement_samples]
        
        ax1.scatter(
            refinement_afhps,
            refinement_returns,
            c="orange",
            s=80,
            zorder=6,
            marker="s",
            label="Return refinement",
            alpha=0.8,
        )

    # Show bin boundaries
    for edge in bs_sampler.bin_edges:
        ax1.axvline(edge, color="gray", linestyle="--", alpha=0.3)

    ax1.set_xlabel("AFHP (%)")
    ax1.set_ylabel("Simulated Return")
    ax1.set_title(
        f"Coverage Comparison\nEnhanced Binary: {bs_summary['coverage_percentage']:.1f}%, Uniform: {uniform_coverage:.1f}%"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Bin filling visualization
    ax2 = plt.subplot(2, 3, 2)

    # Show which bins were filled by each method
    bin_centers = [
        (bs_sampler.bin_edges[i] + bs_sampler.bin_edges[i + 1]) / 2
        for i in range(num_bins)
    ]
    bin_width = bs_sampler.bin_edges[1] - bs_sampler.bin_edges[0]

    # Binary search coverage
    bs_filled = [1 if s is not None else 0 for s in bs_samples]
    ax2.bar(
        bin_centers,
        bs_filled,
        width=bin_width * 0.4,
        alpha=0.7,
        label="Binary Search",
        color="red",
        align="edge",
    )

    # Uniform coverage
    uniform_filled = [0] * num_bins
    for _, afhp in uniform_samples:
        bin_idx = int(afhp // 10) if afhp < 100 else 9
        uniform_filled[bin_idx] = 1
    ax2.bar(
        [c + bin_width * 0.4 for c in bin_centers],
        uniform_filled,
        width=bin_width * 0.4,
        alpha=0.7,
        label="Uniform",
        color="green",
        align="edge",
    )

    # Add bin edges
    for edge in bs_sampler.bin_edges:
        ax2.axvline(edge, color="gray", linestyle="--", alpha=0.3)

    ax2.set_xlabel("AFHP (%)")
    ax2.set_ylabel("Bin Filled")
    ax2.set_title("Which Bins Were Filled")
    ax2.set_ylim(0, 1.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: True function with sampling order
    ax3 = plt.subplot(2, 3, 3)

    # Plot true function
    x_true = np.linspace(0, 100, 1000)
    y_true = [example_monotonic_function(x) for x in x_true]
    ax3.plot(x_true, y_true, "g-", alpha=0.3, linewidth=2, label="True function")

    # Overlay samples
    all_samples = bs_sampler.get_all_samples()
    x_samples = [s.input_value for s in all_samples]
    y_samples = [s.output_value for s in all_samples]

    # Number them by evaluation order
    for i, (x, y) in enumerate(zip(x_samples, y_samples)):
        ax3.scatter(x, y, c="red", s=100, zorder=5)
        ax3.annotate(
            str(i + 1),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax3.set_xlabel("Input (Percentile)")
    ax3.set_ylabel("Output (AFHP %)")
    ax3.set_title("Sampling Order on True Function")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Sampling order comparison
    ax4 = plt.subplot(2, 3, 4)

    # Binary search sampling order
    all_samples = bs_sampler.get_all_samples()
    eval_numbers = list(range(1, len(all_samples) + 1))
    output_values = [s.output_value for s in all_samples]

    ax4.scatter(eval_numbers, output_values, c="blue", alpha=0.6, label="Binary Search")
    ax4.plot(eval_numbers, output_values, "b-", alpha=0.3)

    # Uniform sampling order
    uniform_eval_numbers = list(range(1, len(uniform_sample_data) + 1))
    uniform_output_values = [afhp for afhp, _ in uniform_sample_data]

    ax4.scatter(
        uniform_eval_numbers,
        uniform_output_values,
        c="green",
        alpha=0.6,
        label="Uniform",
    )
    ax4.plot(uniform_eval_numbers, uniform_output_values, "g--", alpha=0.3)

    ax4.set_xlabel("Evaluation Number")
    ax4.set_ylabel("Output Value (AFHP %)")
    ax4.set_title("Sampling Order Comparison")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Coverage evolution
    ax5 = plt.subplot(2, 3, 5)

    # Binary search coverage evolution
    bs_coverage_pcts = []
    for i in range(1, len(all_samples) + 1):
        bins_filled = set()
        for sample in all_samples[:i]:
            bin_idx = int(sample.output_value // (100 / num_bins))
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            bins_filled.add(bin_idx)
        bs_coverage_pcts.append(100.0 * len(bins_filled) / num_bins)

    ax5.plot(eval_numbers, bs_coverage_pcts, "b-", linewidth=2, label="Binary Search")

    # Uniform sampling coverage evolution
    uniform_coverage_pcts = []
    for i in range(1, len(uniform_sample_data) + 1):
        bins_filled = set()
        for afhp, _ in uniform_sample_data[:i]:
            bin_idx = int(afhp // (100 / num_bins))
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            bins_filled.add(bin_idx)
        uniform_coverage_pcts.append(100.0 * len(bins_filled) / num_bins)

    ax5.plot(
        uniform_eval_numbers,
        uniform_coverage_pcts,
        "g--",
        linewidth=2,
        label="Uniform Sampling",
    )

    ax5.set_xlabel("Evaluation Number")
    ax5.set_ylabel("Coverage (%)")
    ax5.set_title("Coverage Evolution")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 105])
    ax5.legend()

    plt.suptitle("Enhanced Binary Search vs Uniform Sampling Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run basic example
    # sampler, summary = run_example()

    # # Run enhanced example with return gap filling
    # print("\n" + "=" * 80)
    # enhanced_sampler, enhanced_summary = run_enhanced_example()

    # Run comparison
    compare_with_uniform_sampling()

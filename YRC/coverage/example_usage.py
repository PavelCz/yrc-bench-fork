"""
Example usage of the binary search coverage sampler.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

from .binary_search import BinarySearchSampler
from .visualization import plot_coverage_results, plot_sampling_efficiency, create_coverage_report


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
    
    # Simulate return as a function of AFHP
    # More help → better return, but with diminishing returns
    base_return = 50  # Weak agent alone
    max_return = 90   # Strong agent alone
    
    # Logarithmic increase in return with AFHP
    return_value = base_return + (max_return - base_return) * np.log(1 + afhp) / np.log(101)
    return_value += np.random.normal(0, 2)  # Add noise
    
    metadata = {
        "return_mean": return_value,
        "return_std": np.random.uniform(1, 3),
        "threshold_used": threshold,
    }
    
    return afhp, metadata


def run_example():
    """Run a complete example of the binary search sampler."""
    
    print("Binary Search Coverage Sampler Example")
    print("=" * 50)
    
    # Create sampler
    num_bins = 10
    sampler = BinarySearchSampler(
        eval_function=example_evaluation_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
    )
    
    # Run sampling
    print(f"\nRunning sampler with {num_bins} bins...")
    samples = sampler.run()
    
    # Get summary
    summary = sampler.get_coverage_summary()
    print(f"\nSampling complete!")
    print(f"Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"Evaluations: {summary['total_evaluations']}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Coverage results
    fig1 = plot_coverage_results(
        samples,
        bin_edges=sampler.bin_edges,
        title="Binary Search Coverage Sampling Results",
        ylabel="Simulated Return"
    )
    
    # Plot 2: Sampling efficiency
    fig2 = plot_sampling_efficiency(
        sampler.get_all_samples(),
        title="Sampling Efficiency Analysis"
    )
    
    # Generate report
    print("\nCoverage Report:")
    print("-" * 50)
    report = create_coverage_report(sampler)
    print(report)
    
    # Show true function for comparison
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true function
    x_true = np.linspace(0, 100, 1000)
    y_true = [example_monotonic_function(x) for x in x_true]
    ax.plot(x_true, y_true, 'g-', alpha=0.3, linewidth=2, label='True function')
    
    # Overlay samples
    all_samples = sampler.get_all_samples()
    x_samples = [s.input_value for s in all_samples]
    y_samples = [s.output_value for s in all_samples]
    
    # Number them by evaluation order
    for i, (x, y) in enumerate(zip(x_samples, y_samples)):
        ax.scatter(x, y, c='red', s=100, zorder=5)
        ax.annotate(str(i+1), (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Input (Percentile)")
    ax.set_ylabel("Output (AFHP %)")
    ax.set_title("Sampling Order on True Function")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.show()
    
    return sampler, summary


def compare_with_uniform_sampling():
    """Compare binary search with uniform sampling."""
    
    print("\nComparing Binary Search vs Uniform Sampling")
    print("=" * 50)
    
    num_bins = 10
    num_evals = 12  # Slightly more than num_bins
    
    # Run binary search sampler
    bs_sampler = BinarySearchSampler(
        eval_function=example_evaluation_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
    )
    bs_samples = bs_sampler.run()
    bs_summary = bs_sampler.get_coverage_summary()
    
    # Run uniform sampling for comparison
    uniform_x = np.linspace(0, 100, num_evals)
    uniform_samples = []
    for x in uniform_x:
        afhp, metadata = example_evaluation_function(x)
        uniform_samples.append((x, afhp))
    
    # Compute uniform coverage
    uniform_bins_filled = set()
    for _, afhp in uniform_samples:
        bin_idx = int(afhp // 10) if afhp < 100 else 9
        uniform_bins_filled.add(bin_idx)
    
    uniform_coverage = 100.0 * len(uniform_bins_filled) / num_bins
    
    print(f"\nBinary Search:")
    print(f"  Coverage: {bs_summary['coverage_percentage']:.1f}%")
    print(f"  Evaluations: {bs_summary['total_evaluations']}")
    
    print(f"\nUniform Sampling:")
    print(f"  Coverage: {uniform_coverage:.1f}%")
    print(f"  Evaluations: {num_evals}")
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Binary search results
    valid_samples = [s for s in bs_samples if s is not None]
    ax1.scatter([s.output_value for s in valid_samples], 
                [s.metadata["return_mean"] for s in valid_samples],
                c='red', s=100, label='Samples')
    
    for edge in bs_sampler.bin_edges:
        ax1.axvline(edge, color='gray', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel("AFHP (%)")
    ax1.set_ylabel("Return")
    ax1.set_title(f"Binary Search (Coverage: {bs_summary['coverage_percentage']:.1f}%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Uniform sampling results
    ax2.scatter([afhp for _, afhp in uniform_samples],
                [50 + 40 * np.log(1 + afhp) / np.log(101) for _, afhp in uniform_samples],
                c='blue', s=100, label='Samples')
    
    for edge in np.linspace(0, 100, num_bins + 1):
        ax2.axvline(edge, color='gray', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel("AFHP (%)")
    ax2.set_ylabel("Return")
    ax2.set_title(f"Uniform Sampling (Coverage: {uniform_coverage:.1f}%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run basic example
    sampler, summary = run_example()
    
    # Run comparison
    compare_with_uniform_sampling()
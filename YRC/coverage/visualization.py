"""
Visualization utilities for coverage algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from .binary_search import SamplePoint


def plot_coverage_results(
    samples: List[SamplePoint],
    bin_edges: Optional[np.ndarray] = None,
    title: str = "Coverage Sampling Results",
    xlabel: str = "Ask for Help Percentage (%)",
    ylabel: str = "Return",
    figsize: Tuple[int, int] = (10, 6),
    show_bins: bool = True,
    show_order: bool = False,
) -> plt.Figure:
    """
    Plot the results of coverage sampling.
    
    Args:
        samples: List of sample points (can include None values)
        bin_edges: Optional array of bin edges for visualization
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        show_bins: Whether to show bin boundaries
        show_order: Whether to number points by evaluation order
        
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out None samples
    valid_samples = [s for s in samples if s is not None]
    
    if not valid_samples:
        ax.text(0.5, 0.5, "No valid samples", ha='center', va='center')
        return fig
    
    # Extract data
    output_values = [s.output_value for s in valid_samples]
    returns = [s.metadata.get('summary', {}).get('return_mean', 0) for s in valid_samples]
    
    # Sort by output value for line plot
    sorted_indices = np.argsort(output_values)
    sorted_outputs = [output_values[i] for i in sorted_indices]
    sorted_returns = [returns[i] for i in sorted_indices]
    
    # Plot the curve
    ax.plot(sorted_outputs, sorted_returns, 'b-', linewidth=2, label='Sampled curve')
    ax.scatter(output_values, returns, c='red', s=50, zorder=5, label='Sample points')
    
    # Show evaluation order if requested
    if show_order:
        for i, sample in enumerate(valid_samples):
            if sample is not None:
                ax.annotate(
                    str(i),
                    (sample.output_value, sample.metadata.get('summary', {}).get('return_mean', 0)),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                )
    
    # Show bin boundaries if provided
    if show_bins and bin_edges is not None:
        for edge in bin_edges:
            ax.axvline(edge, color='gray', linestyle='--', alpha=0.3)
        
        # Highlight empty bins
        for i in range(len(samples)):
            if samples[i] is None and i < len(bin_edges) - 1:
                ax.axvspan(
                    bin_edges[i], bin_edges[i + 1],
                    color='red', alpha=0.1,
                    label='Empty bin' if i == 0 else None
                )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def plot_sampling_efficiency(
    all_samples: List[SamplePoint],
    title: str = "Sampling Efficiency",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot how the coverage evolves with each evaluation.
    
    Args:
        all_samples: All samples in evaluation order
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Track coverage evolution
    eval_numbers = list(range(1, len(all_samples) + 1))
    output_values = [s.output_value for s in all_samples]
    
    # Plot 1: Output values over evaluations
    ax1.scatter(eval_numbers, output_values, c='blue', alpha=0.6)
    ax1.plot(eval_numbers, output_values, 'b-', alpha=0.3)
    ax1.set_xlabel("Evaluation Number")
    ax1.set_ylabel("Output Value (AFHP %)")
    ax1.set_title("Sampling Order")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage percentage over evaluations
    coverage_pcts = []
    output_ranges = []
    
    for i in range(1, len(all_samples) + 1):
        current_outputs = [s.output_value for s in all_samples[:i]]
        if current_outputs:
            output_range = max(current_outputs) - min(current_outputs)
            output_ranges.append(output_range)
            # Simple coverage metric: range covered / total possible range
            coverage_pcts.append(output_range / 100.0 * 100)
    
    ax2.plot(eval_numbers, coverage_pcts, 'g-', linewidth=2)
    ax2.set_xlabel("Evaluation Number")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Coverage Evolution")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def create_coverage_report(
    sampler,
    save_path: Optional[str] = None,
) -> str:
    """
    Create a text report of the coverage results.
    
    Args:
        sampler: BinarySearchSampler instance after running
        save_path: Optional path to save the report
        
    Returns:
        Report as a string
    """
    summary = sampler.get_coverage_summary()
    filled_samples = sampler.get_filled_samples()
    
    report_lines = [
        "=" * 60,
        "Coverage Sampling Report",
        "=" * 60,
        f"Total bins: {sampler.num_bins}",
        f"Bins filled: {summary['bins_filled']}",
        f"Coverage percentage: {summary['coverage_percentage']:.1f}%",
        f"Total evaluations: {summary['total_evaluations']}",
        f"Efficiency: {summary['coverage_percentage'] / summary['total_evaluations']:.2f}% coverage per eval",
        "",
        "Output range covered: {:.2f} to {:.2f}".format(*summary['output_range_covered']),
        "",
    ]
    
    if summary['gaps']:
        report_lines.extend([
            "Gaps in coverage:",
            "-" * 30,
        ])
        for i, (start, end) in enumerate(summary['gaps']):
            report_lines.append(f"  Gap {i+1}: [{start:.1f}, {end:.1f}]")
        report_lines.append("")
    
    # Sample details
    report_lines.extend([
        "Filled bins:",
        "-" * 30,
    ])
    
    for i, sample in enumerate(sampler.bin_samples):
        if sample is not None:
            report_lines.append(
                f"  Bin {i}: Output={sample.output_value:.2f}, "
                f"Input={sample.input_value:.2f}"
            )
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report
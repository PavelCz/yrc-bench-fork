#!/usr/bin/env python3
"""
Common plotting configuration and utilities shared between plotting scripts.
"""

import matplotlib.pyplot as plt
from typing import List, Optional

# Method display names mapping
METHOD_NAMES = {
    "max_prob": r"\textsc{MaxProb}",
    "max_logit": r"\textsc{MaxLogit}",
    "lb_random": r"\textsc{Level-Based Random}",
    "ts_random": r"\textsc{Heuristic}",
    "svdd_image": r"\textsc{ImageSVDD}",
    "svdd_latent": r"\textsc{LatentSVDD}",
    "ensemble": r"\textsc{Ensemble (multi)}",
    # Ensemble Variance (Single Weak)
    "ensemble_single": r"\textsc{Ensemble}",
    "latent-svdd": r"\textsc{Latent SVDD}",
    # "random": "Timestep Random",
    "oc-random": r"\textsc{Level-Based Random}",
    "oracle_lb_random": r"\textsc{PartialOracle}",
    "wait": r"\textsc{Wait}",
}


def setup_plot_style(paper_mode: bool = False, use_latex: bool = True) -> None:
    """
    Set up common plot styling for all plots.
    
    Args:
        paper_mode: If True, prepare for paper publication
        use_latex: If True, use LaTeX rendering for text
    """
    if use_latex:
        # Configure matplotlib to use LaTeX rendering
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'  # Palatino font in LaTeX
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino Linotype', 'Palatino', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Increase font sizes
    plt.rcParams['font.size'] = 12  # Base font size
    plt.rcParams['axes.labelsize'] = 16  # Axis labels
    plt.rcParams['axes.titlesize'] = 16  # Title (if used)
    plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 12  # Legend text


def get_line_styles(num_methods: int, paper_mode: bool = False, method_names: Optional[List[str]] = None) -> List[str]:
    """
    Get line styles for plotting multiple methods.
    
    Args:
        num_methods: Number of methods to plot
        paper_mode: If True, return different line styles for distinction in B&W
        method_names: Optional list of method names to handle special cases
        
    Returns:
        List of line style specifications
    """
    if paper_mode:
        # Define all available unique line styles
        all_line_styles = [
            '-',      # solid
            '--',     # dashed
            '-.',     # dash-dot
            ':',      # dotted
            (0, (3, 1, 1, 1)),  # densely dashdotted
            (0, (5, 1)),        # densely dashed
            (0, (1, 1)),        # densely dotted
            (0, (3, 5, 1, 5)),  # dashdotdotted
            (0, (5, 5)),        # long dash with offset
            (0, (3, 1, 1, 1, 1, 1)),  # dashdotdotted variant
            (0, (1, 5)),        # dotted with long gaps
            (0, (5, 10)),       # long dash with very long gaps
        ]
        
        # Simply assign styles in order - each method gets a unique style
        if num_methods <= len(all_line_styles):
            return all_line_styles[:num_methods]
        else:
            # If we need more styles, cycle through them
            styles = []
            for i in range(num_methods):
                styles.append(all_line_styles[i % len(all_line_styles)])
            return styles
    else:
        # All solid lines when not in paper mode
        return ['-'] * num_methods


def style_plot_for_publication(
    ax=None,
    legend_outside: bool = True,
    legend_location: str = 'center left',
    legend_bbox_to_anchor: tuple = (1.05, 0.5),
    handles=None,
    labels=None,
) -> None:
    """
    Apply publication-ready styling to current plot.

    Args:
        ax: Matplotlib axes object (uses current axes if None)
        legend_outside: If True, place legend outside plot area
        legend_location: Legend location string
        legend_bbox_to_anchor: Anchor point for legend when outside
        handles: Optional explicit legend handles. When provided alongside
            `labels`, the legend uses them instead of auto-discovering from
            the current axes.
        labels: Optional explicit legend labels; must accompany `handles`.
    """
    if ax is None:
        ax = plt.gca()

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_kwargs = {"frameon": True, "fancybox": False}
    if legend_outside:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        legend_kwargs["loc"] = legend_location
    else:
        legend_kwargs["loc"] = "best"

    if handles is not None and labels is not None:
        legend = plt.legend(handles, labels, **legend_kwargs)
    else:
        legend = plt.legend(**legend_kwargs)

    # White background, no border
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('none')

    # Add grid
    plt.grid(True, alpha=0.3)


def format_label(method: str, paper_mode: bool, n_experiments: Optional[int] = None) -> str:
    """
    Format method label for legend.
    
    Args:
        method: Method name
        paper_mode: If True, exclude n= information
        n_experiments: Number of experiments (if aggregating)
        
    Returns:
        Formatted label string
    """
    # Get display name
    label = METHOD_NAMES.get(method, method)
    
    # Add experiment count if not in paper mode
    if not paper_mode and n_experiments is not None:
        label = f"{label} (n={n_experiments})"
    
    return label
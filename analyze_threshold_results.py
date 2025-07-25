import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_results(filepath: str) -> Dict:
    """Load results from npz file"""
    try:
        data = np.load(filepath, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def analyze_coverage(afhp_values: np.ndarray, returns: np.ndarray, method_name: str = ""):
    """Analyze the coverage quality of the sampling"""
    print(f"\n=== Coverage Analysis for {method_name} ===")
    print(f"Number of points: {len(afhp_values)}")
    print(f"AFHP range: [{np.min(afhp_values):.3f}, {np.max(afhp_values):.3f}]")
    print(f"Return range: [{np.min(returns):.3f}, {np.max(returns):.3f}]")
    
    # Calculate gaps in AFHP dimension
    sorted_afhp = np.sort(afhp_values)
    afhp_gaps = np.diff(sorted_afhp)
    print(f"AFHP gaps - Mean: {np.mean(afhp_gaps):.4f}, Max: {np.max(afhp_gaps):.4f}, Std: {np.std(afhp_gaps):.4f}")
    
    # Calculate gaps in return dimension (sorted by AFHP)
    sorted_indices = np.argsort(afhp_values)
    sorted_returns = returns[sorted_indices]
    return_gaps = np.abs(np.diff(sorted_returns))
    print(f"Return gaps - Mean: {np.mean(return_gaps):.4f}, Max: {np.max(return_gaps):.4f}, Std: {np.std(return_gaps):.4f}")
    
    # Calculate uniformity metrics
    afhp_uniformity = np.std(afhp_gaps) / np.mean(afhp_gaps) if np.mean(afhp_gaps) > 0 else 0
    return_uniformity = np.std(return_gaps) / np.mean(return_gaps) if np.mean(return_gaps) > 0 else 0
    print(f"Uniformity (lower=better) - AFHP: {afhp_uniformity:.4f}, Return: {return_uniformity:.4f}")
    
    return {
        'afhp_gaps': afhp_gaps,
        'return_gaps': return_gaps,
        'afhp_uniformity': afhp_uniformity,
        'return_uniformity': return_uniformity
    }


def plot_results(results_dict: Dict[str, Dict], save_path: str = None):
    """Create comprehensive plots comparing different approaches"""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Colors for different methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    method_colors = dict(zip(results_dict.keys(), colors))
    
    # Plot 1: AFHP vs Returns scatter plot
    ax1 = fig.add_subplot(gs[0, :2])
    for method_name, data in results_dict.items():
        afhp_values = data['afhp_values']
        returns = data['returns']
        ax1.scatter(afhp_values, returns, alpha=0.7, s=60, 
                   label=f'{method_name} (n={len(afhp_values)})',
                   color=method_colors[method_name])
    
    ax1.set_xlabel('Ask for Help Percentage (AFHP)')
    ax1.set_ylabel('Average Return')
    ax1.set_title('AFHP vs Returns Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AFHP vs Returns with connecting lines
    ax2 = fig.add_subplot(gs[0, 2:])
    for method_name, data in results_dict.items():
        afhp_values = data['afhp_values']
        returns = data['returns']
        # Sort by AFHP for line plot
        sorted_indices = np.argsort(afhp_values)
        sorted_afhp = afhp_values[sorted_indices]
        sorted_returns = returns[sorted_indices]
        
        ax2.plot(sorted_afhp, sorted_returns, 'o-', alpha=0.7, 
                label=method_name, color=method_colors[method_name], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Ask for Help Percentage (AFHP)')
    ax2.set_ylabel('Average Return')
    ax2.set_title('AFHP vs Returns (Sorted by AFHP)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: AFHP gap distribution
    ax3 = fig.add_subplot(gs[1, 0])
    for method_name, data in results_dict.items():
        afhp_values = data['afhp_values']
        sorted_afhp = np.sort(afhp_values)
        afhp_gaps = np.diff(sorted_afhp)
        ax3.hist(afhp_gaps, alpha=0.6, bins=15, label=method_name, 
                color=method_colors[method_name], density=True)
    
    ax3.set_xlabel('AFHP Gap Size')
    ax3.set_ylabel('Density')
    ax3.set_title('AFHP Gap Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Return gap distribution
    ax4 = fig.add_subplot(gs[1, 1])
    for method_name, data in results_dict.items():
        afhp_values = data['afhp_values']
        returns = data['returns']
        sorted_indices = np.argsort(afhp_values)
        sorted_returns = returns[sorted_indices]
        return_gaps = np.abs(np.diff(sorted_returns))
        ax4.hist(return_gaps, alpha=0.6, bins=15, label=method_name, 
                color=method_colors[method_name], density=True)
    
    ax4.set_xlabel('Return Gap Size')
    ax4.set_ylabel('Density')
    ax4.set_title('Return Gap Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Threshold distribution
    ax5 = fig.add_subplot(gs[1, 2])
    for method_name, data in results_dict.items():
        thresholds = data['thresholds']
        # Filter out infinite values for histogram
        finite_thresholds = thresholds[np.isfinite(thresholds)]
        if len(finite_thresholds) > 0:
            ax5.hist(finite_thresholds, alpha=0.6, bins=15, label=method_name, 
                    color=method_colors[method_name], density=True)
    
    ax5.set_xlabel('Threshold Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Threshold Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative AFHP coverage
    ax6 = fig.add_subplot(gs[1, 3])
    for method_name, data in results_dict.items():
        afhp_values = data['afhp_values']
        sorted_afhp = np.sort(afhp_values)
        cumulative = np.arange(1, len(sorted_afhp) + 1) / len(sorted_afhp)
        ax6.plot(sorted_afhp, cumulative, 'o-', alpha=0.7, 
                label=method_name, color=method_colors[method_name])
    
    ax6.set_xlabel('AFHP Value')
    ax6.set_ylabel('Cumulative Fraction')
    ax6.set_title('Cumulative AFHP Coverage')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7 & 8: Summary statistics
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Prepare data for bar plot
    methods = list(results_dict.keys())
    afhp_uniformity = []
    return_uniformity = []
    num_points = []
    
    for method_name in methods:
        data = results_dict[method_name]
        afhp_values = data['afhp_values']
        returns = data['returns']
        
        # Calculate uniformity metrics
        sorted_afhp = np.sort(afhp_values)
        afhp_gaps = np.diff(sorted_afhp)
        sorted_indices = np.argsort(afhp_values)
        sorted_returns = returns[sorted_indices]
        return_gaps = np.abs(np.diff(sorted_returns))
        
        afhp_unif = np.std(afhp_gaps) / np.mean(afhp_gaps) if np.mean(afhp_gaps) > 0 else 0
        return_unif = np.std(return_gaps) / np.mean(return_gaps) if np.mean(return_gaps) > 0 else 0
        
        afhp_uniformity.append(afhp_unif)
        return_uniformity.append(return_unif)
        num_points.append(len(afhp_values))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, afhp_uniformity, width, label='AFHP Uniformity', alpha=0.7)
    bars2 = ax7.bar(x + width/2, return_uniformity, width, label='Return Uniformity', alpha=0.7)
    
    ax7.set_xlabel('Method')
    ax7.set_ylabel('Uniformity (lower is better)')
    ax7.set_title('Uniformity Comparison')
    ax7.set_xticks(x)
    ax7.set_xticklabels(methods, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax7.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax7.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 8: Number of evaluations
    ax8 = fig.add_subplot(gs[2, 2:])
    bars = ax8.bar(methods, num_points, alpha=0.7, color=[method_colors[m] for m in methods])
    ax8.set_ylabel('Number of Evaluations')
    ax8.set_title('Evaluation Efficiency')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax8.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.suptitle('Threshold Evaluation Results Comparison', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze threshold evaluation results')
    parser.add_argument('files', nargs='+', help='Path to npz result files')
    parser.add_argument('--save', type=str, help='Path to save the plot')
    args = parser.parse_args()
    
    results_dict = {}
    
    for filepath in args.files:
        data = load_results(filepath)
        if data is not None:
            # Extract method name from filename
            method_name = Path(filepath).stem
            
            # Convert to numpy arrays
            afhp_values = np.array(data['afhp_values'])
            returns = np.array(data['returns'])
            thresholds = np.array(data['thresholds'])
            
            results_dict[method_name] = {
                'afhp_values': afhp_values,
                'returns': returns,
                'thresholds': thresholds
            }
            
            # Analyze coverage for this method
            analyze_coverage(afhp_values, returns, method_name)
    
    if results_dict:
        # Create comparison plots
        plot_results(results_dict, args.save)
        
        print("\n=== Summary ===")
        print("Lower uniformity values indicate more even spacing.")
        print("The ideal approach should have:")
        print("1. Good coverage across both AFHP and return dimensions")
        print("2. Low uniformity scores (even spacing)")
        print("3. Reasonable number of evaluations")
    else:
        print("No valid result files found.")


if __name__ == "__main__":
    main() 
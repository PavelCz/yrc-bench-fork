# Threshold Evaluation Approaches

This guide explains different approaches for evaluating AE-based policies across threshold values to ensure good coverage across both AFHP (Ask for Help Percentage) and return dimensions.

## Problem Statement

The original AFHP-based binning approach only considers spacing along the AFHP axis, which can lead to:
- Uneven spacing in the return dimension
- Sudden jumps in plots when visualizing returns vs AFHP
- Poor coverage of the actual performance trade-off space

**Key insight**: We cannot directly target specific AFHP values since the actual AFHP is only known after evaluation on the test set. The `train_percentile` method gives us thresholds based on training data, but the empirical AFHP on test data may be quite different. Therefore, we must sample thresholds systematically and then adaptively fill gaps based on observed results.

## Proposed Solutions

### Option 1: Two-Phase Approach (Recommended)
**File**: `eval_thresholds.py` (modified original)

This approach uses:
1. **Phase 1**: Systematic threshold sampling across training percentiles
2. **Phase 2**: Adaptive gap filling based on observed (AFHP, return) results

**Advantages**:
- Clean and simple implementation
- Builds on existing code structure
- Provides good coverage across both dimensions
- Computationally efficient

**Usage**:
```bash
python eval_thresholds.py --config your_config.yaml --eval.threshold_bins 20
```

### Option 2: Advanced 2D Adaptive Sampling
**File**: `eval_thresholds_advanced.py`

This approach uses:
1. **Extreme point evaluation** to establish bounds
2. **Stratified threshold sampling** across training percentiles with randomization
3. **Adaptive 2D refinement** based on gaps in observed (AFHP, return) space

**Advantages**:
- More sophisticated sampling strategy
- Better theoretical coverage guarantees
- Adaptive to the actual trade-off landscape
- Includes method tracking for analysis

**Usage**:
```bash
python eval_thresholds_advanced.py --config your_config.yaml --eval.threshold_bins 20
```

## Analysis and Visualization

### Analyzing Results
**File**: `analyze_threshold_results.py`

Use this script to compare different approaches and analyze coverage quality:

```bash
python analyze_threshold_results.py results_basic_test.npz results_advanced_test.npz --save comparison_plot.png
```

**Features**:
- Coverage analysis with uniformity metrics
- Comprehensive visualization plots
- Gap distribution analysis
- Method comparison

### Key Metrics

1. **Uniformity Score**: Standard deviation of gaps / Mean gap size
   - Lower values = more even spacing
   - Calculated for both AFHP and return dimensions

2. **Coverage Range**: Min and max values for AFHP and returns

3. **Gap Distribution**: Histogram of gaps between consecutive points

## Implementation Details

### Key Functions

1. **`systematic_threshold_sampling(...)`**: 
   - Samples thresholds systematically across training percentiles
   - Uses stratified sampling for better coverage
   - Records both training percentile and observed AFHP

2. **`adaptive_gap_filling(...)`**:
   - Identifies largest gaps in 2D (AFHP, return) space
   - Uses interpolation of training percentiles when possible
   - Falls back to unexplored percentiles

3. **`adaptive_refinement_advanced(...)`**:
   - Uses 2D distance metrics to find under-sampled regions
   - Employs Voronoi-like approach for gap detection
   - Learns from observed AFHP-percentile mappings

### Data Structure

Results are saved with the following structure:
```python
{
    'thresholds': [...],        # Threshold values used
    'summaries': [...],         # Full evaluation summaries
    'afhp_values': [...],       # Ask for help percentages
    'returns': [...],           # Average returns
    'methods': [...],           # Method used for each point (advanced only)
}
```

## Recommendations

### For Most Use Cases
Use **Option 1 (Two-Phase Approach)**:
- Simpler to understand and debug
- Good coverage with reasonable computational cost
- Builds on existing infrastructure

### For Research/Advanced Analysis
Use **Option 2 (Advanced 2D Adaptive)**:
- More principled sampling approach
- Better for detailed analysis of trade-off landscapes
- Includes method tracking for understanding sampling behavior

### Configuration Tips

1. **Number of threshold bins**: Start with 15-20 for good coverage
2. **Gap filling budget**: Use ~50% of total budget for gap filling in two-phase approach
3. **Analysis**: Always run the visualization script to check coverage quality

## Example Workflow

1. **Run evaluation**:
   ```bash
   python eval_thresholds.py --config config.yaml --eval.threshold_bins 15
   ```

2. **Analyze results**:
   ```bash
   python analyze_threshold_results.py results_test.npz
   ```

3. **Compare approaches** (optional):
   ```bash
   python eval_thresholds_advanced.py --config config.yaml --eval.threshold_bins 15
   python analyze_threshold_results.py results_test.npz results_advanced_test.npz --save comparison.png
   ```

4. **Check for good coverage**:
   - Low uniformity scores (< 1.0 ideally)
   - Smooth curves in AFHP vs return plots
   - No large gaps in either dimension 
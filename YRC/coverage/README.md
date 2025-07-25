# Coverage Algorithms for Monotonic Curve Sampling

This module contains algorithms for efficiently sampling points along monotonic curves, particularly useful for threshold evaluation in coordination policies.

## Overview

When evaluating threshold-based policies, we need to understand how performance (return) varies with the ask-for-help percentage (AFHP). This creates a monotonic curve where:
- **X-axis**: AFHP (0% to 100%)
- **Y-axis**: Return/Performance

The task is to sample thresholds to characterize this curve with a limited evaluation budget.

## Algorithm: Adaptive Binary Search with Binning

The implementation uses a binary search algorithm that fills bins along the AFHP axis. This approach applies when:
1. The goal is uniform coverage along the X-axis (AFHP)
2. The relationship between threshold percentiles and AFHP is monotonic
3. The evaluation results may contain noise

### Key Concepts

1. **Threshold Percentiles**: Training data percentiles that determine thresholds
   - 0th percentile → highest threshold → never ask for help (0% AFHP)
   - 100th percentile → lowest threshold → always ask for help (100% AFHP)

2. **AFHP Bins**: Fixed bins along the X-axis (e.g., [0-10%], [10-20%], ..., [90-100%])

3. **Binary Search**: Recursively bisect the percentile space to fill empty bins

### Algorithm Steps

1. **Initialize**: Evaluate extreme cases (0% and 100% AFHP)
2. **Binary Search**: 
   - Find the middle percentile between left and right bounds
   - Evaluate at that percentile to get actual AFHP
   - Determine which bin the AFHP falls into
   - Recursively search left and right halves if bins remain empty
3. **Termination**: Stop when all bins are filled or no more evaluations possible

### Advantages

- **Efficient**: Uses binary search to minimize evaluations
- **Adaptive**: Focuses on filling empty bins
- **Robust**: Handles non-linear percentile-to-AFHP mappings
- **Predictable**: Guarantees coverage across the entire AFHP range

### Limitations

- **Bin-based**: May miss interesting features between bins
- **Fixed resolution**: Number of bins determines curve resolution
- **Assumes monotonicity**: Not suitable for non-monotonic relationships

## Future Improvements

For better curve characterization, consider:

1. **Adaptive Resolution**: Variable bin sizes based on curve steepness
2. **Curvature-based Sampling**: Focus on areas of high curvature
3. **Confidence Intervals**: Account for evaluation noise
4. **Multi-objective**: Consider both AFHP coverage and return variance

## Implementation

See `binary_search.py` for the generic implementation of this algorithm.
# Coverage Algorithms for Monotonic Curve Sampling

This module contains algorithms for efficiently sampling points along monotonic curves, particularly useful for threshold evaluation in coordination policies.

## Overview

When evaluating threshold-based policies, we need to understand how performance (return) varies with the ask-for-help percentage (AFHP). This creates a monotonic curve where:
- **X-axis**: AFHP (0% to 100%)
- **Y-axis**: Return/Performance

The task is to sample thresholds to characterize this curve with a limited evaluation budget.

## Algorithm: Adaptive Coverage Sampling (ABCS)

The ABCS library now provides a single-phase JointCoverageSampler that adaptively fills gaps to ensure coverage across both AFHP (x) and return (y).

### When to Use ABCS

This approach applies when:
1. The goal is uniform coverage along the X-axis (AFHP)
2. The relationship between threshold percentiles and AFHP is monotonic
3. The evaluation results may contain noise
4. You need to characterize the full trade-off space between control frequency and performance

### Key Concepts

1. **Threshold Percentiles**: Training data percentiles that determine thresholds
   - 0th percentile → highest threshold → never ask for help (0% AFHP)
   - 100th percentile → lowest threshold → always ask for help (100% AFHP)

2. **AFHP Bins**: Fixed bins along the X-axis (e.g., [0-10%], [10-20%], ..., [90-100%])

3. **Binary Search**: Recursively bisect the percentile space to fill empty bins

### Single-phase Joint Coverage

1. Evaluate the extremes (0% and 100% AFHP) to seed the curve
2. Iteratively split the largest normalized neighbor gap on either AFHP or return axis
3. Re-run adjacent pairs that violate monotonicity due to noise
4. Stop when both axes meet the target coverage fraction or the evaluation budget is exhausted

### Advantages (updated)

- **Efficient**: Uses binary search to minimize evaluations needed for full coverage
- **Adaptive**: Automatically identifies and fills coverage gaps on both axes in a single phase
- **Comprehensive**: Ensures good coverage on both performance dimensions (AFHP and return)
- **Deterministic**: Provides consistent, reproducible results
- **Guaranteed AFHP Coverage**: Phase 1 ensures 100% coverage of AFHP bins when the function spans the full range
- **Robust**: Handles non-linear percentile-to-AFHP mappings and evaluation noise
- **Flexible**: Return refinement phase can be enabled/disabled based on needs

### Limitations

- **Bin-based**: May miss interesting features between bins
- **Fixed resolution**: Number of bins determines curve resolution
- **Assumes monotonicity**: Not suitable for non-monotonic relationships
- **Evaluation budget**: Return refinement quality depends on available evaluations

## Future Improvements

For better curve characterization, consider:

1. **Adaptive Resolution**: Variable bin sizes based on curve steepness
2. **Curvature-based Sampling**: Focus on areas of high curvature
3. **Confidence Intervals**: Account for evaluation noise
4. **Multi-objective**: Consider both AFHP coverage and return variance

## Implementation

See `binary_search.py` for the generic implementation of the ABCS algorithm.

### Usage Example (conceptual)

```python
from abcs import JointCoverageSampler

sampler = JointCoverageSampler(
    eval_at_percentile=your_eval_at_percentile,
    eval_at_lower_extreme=your_eval_at_lower_extreme,
    eval_at_upper_extreme=your_eval_at_upper_extreme,
    coverage_fraction=0.10,
    max_total_evals=200,
)
result = sampler.run()
print(result.coverage_x_max_gap, result.coverage_y_max_gap)
```

### Testing

The module includes comprehensive tests (`test_coverage.py`) that verify:
- 100% AFHP coverage guarantee for reasonable bin counts
- 100% return coverage when given sufficient evaluation budget
- Robustness across different parameter configurations

The tests demonstrate that ABCS achieves its coverage guarantees when the evaluation function spans the expected output range.
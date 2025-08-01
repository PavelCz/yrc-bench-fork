# Coverage Algorithms for Monotonic Curve Sampling

This module contains algorithms for efficiently sampling points along monotonic curves, particularly useful for threshold evaluation in coordination policies.

## Overview

When evaluating threshold-based policies, we need to understand how performance (return) varies with the ask-for-help percentage (AFHP). This creates a monotonic curve where:
- **X-axis**: AFHP (0% to 100%)
- **Y-axis**: Return/Performance

The task is to sample thresholds to characterize this curve with a limited evaluation budget.

## Algorithm: Adaptive Binary Coverage Search (ABCS)

The ABCS algorithm is a two-stage adaptive sampling algorithm for efficient coverage of monotonic evaluation curves. It uses binary search to minimize evaluations while ensuring comprehensive coverage across both AFHP and return dimensions.

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

### Algorithm Phases

The ABCS algorithm operates in two distinct phases:

#### Phase 1: AFHP Coverage via Binary Search

1. **Initialize**: Evaluate extreme cases (0% and 100% AFHP)
2. **Binary Search**: 
   - Find the middle percentile between left and right bounds
   - Evaluate at that percentile to get actual AFHP
   - Determine which bin the AFHP falls into
   - Recursively search left and right halves if bins remain empty
3. **Termination**: Stop when all AFHP bins are filled

This phase guarantees 100% coverage of AFHP bins by systematically subdividing the input space.

#### Phase 2: Return Value Refinement (Optional)

1. **Gap Identification**: Analyze return values from Phase 1 to identify gaps
2. **Return Binning**: Create bins along the return axis based on observed range
3. **Targeted Sampling**: Use binary search to find thresholds that produce returns in empty bins
4. **Termination**: Stop when return bins are filled or evaluation budget is exhausted

This phase ensures smooth, well-characterized performance curves across the return dimension.

### Advantages

- **Efficient**: Uses binary search to minimize evaluations needed for full coverage
- **Adaptive**: Automatically identifies and fills coverage gaps on both axes
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

### Usage Example

```python
from YRC.coverage.binary_search import BinarySearchSampler

# Create sampler with both AFHP and return coverage
sampler = BinarySearchSampler(
    eval_function=your_evaluation_function,
    num_bins=10,  # AFHP bins
    return_bins=8,  # Return bins (0 to disable Phase 2)
    max_additional_evals=20,  # Budget for return refinement
    verbose=True
)

# Run the two-phase algorithm
samples = sampler.run_with_return_refinement()

# Get coverage statistics
summary = sampler.get_coverage_summary()
print(f"AFHP coverage: {summary['coverage_percentage']}%")
```

### Testing

The module includes comprehensive tests (`test_coverage.py`) that verify:
- 100% AFHP coverage guarantee for reasonable bin counts
- 100% return coverage when given sufficient evaluation budget
- Robustness across different parameter configurations

The tests demonstrate that ABCS achieves its coverage guarantees when the evaluation function spans the expected output range.
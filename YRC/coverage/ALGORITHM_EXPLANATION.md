# Detailed Algorithm Explanation: Binary Search Coverage Sampling

## Problem Statement

When evaluating threshold-based coordination policies, we need to understand the relationship between:
- **Input**: Threshold percentile (0-100%)
- **Output**: Ask-for-help percentage (AFHP) (0-100%)
- **Performance**: Return/reward achieved

This creates a monotonic curve where lower thresholds (higher percentiles) lead to more frequent help requests.

## The Binary Search Solution

The algorithm uses binary search to efficiently fill bins along the AFHP axis. Here's why this works well:

### 1. **Monotonicity Exploitation**

Since the percentile-to-AFHP mapping is monotonic:
- Lower percentile → Higher threshold → Less help → Lower AFHP
- Higher percentile → Lower threshold → More help → Higher AFHP

This monotonicity enables binary search.

### 2. **Bin-Based Coverage**

The algorithm divides the AFHP range into equal bins (e.g., [0-10%], [10-20%], ..., [90-100%]). The goal is to have at least one sample in each bin, ensuring uniform coverage along the X-axis.

### 3. **Recursive Binary Search**

```
Initial state: Evaluate at 0% and 100% percentiles
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  ✓  │     │     │     │     │     │     │     │     │  ✓  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  0-10  10-20 20-30 30-40 40-50 50-60 60-70 70-80 80-90 90-100

Step 1: Evaluate at 50% percentile (middle)
        Suppose it gives AFHP = 45%
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  ✓  │     │     │     │  ✓  │     │     │     │     │  ✓  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Step 2: Recursively search [0%, 50%] and [50%, 100%]
        Continue until all bins are filled
```

### 4. **Algorithm Walkthrough**

```python
def determine_results(left_percentile, right_percentile, left_bin, right_bin):
    # 1. Calculate middle percentile
    middle_percentile = (left_percentile + right_percentile) / 2
    
    # 2. Convert to threshold and evaluate
    threshold = policy.train_percentile(100 - middle_percentile)
    afhp = evaluate(threshold)
    
    # 3. Determine which bin this AFHP falls into
    bin_idx = determine_bin(afhp)
    
    # 4. Fill the bin if empty
    if bin_is_empty(bin_idx):
        fill_bin(bin_idx, result)
    
    # 5. Recursively search left and right
    if bins_remain(left_bin, bin_idx):
        determine_results(left_percentile, middle_percentile, left_bin, bin_idx)
    
    if bins_remain(bin_idx, right_bin):
        determine_results(middle_percentile, right_percentile, bin_idx, right_bin)
```

## Advantages of This Approach

1. **Efficiency**: O(n log n) evaluations for n bins
2. **Guaranteed Coverage**: Ensures all bins get filled (if possible)
3. **Adaptive**: Automatically handles non-linear percentile-to-AFHP mappings
4. **Simple**: Easy to implement and understand

## Limitations and Improvements

### Current Limitations

1. **Fixed Resolution**: Number of bins is predetermined
2. **Uniform Bins**: All bins have equal width
3. **Single Sample per Bin**: Doesn't capture within-bin variation
4. **No Noise Handling**: Assumes deterministic evaluations

### Potential Improvements

1. **Adaptive Bin Sizing**: Smaller bins where the curve is steep
2. **Multiple Samples per Bin**: Better characterization of each region
3. **Confidence Intervals**: Account for evaluation noise
4. **Interpolation**: Smooth curve fitting between samples

## Example Usage

```python
from YRC.coverage import BinarySearchSampler, create_threshold_sampler

# Create a sampler for threshold evaluation
sampler = create_threshold_sampler(
    policy=policy,
    evaluator=evaluator,
    envs=envs,
    split="test",
    num_bins=20,
    logger=wandb_logger
)

# Run the sampling
samples = sampler.run()

# Get coverage summary
summary = sampler.get_coverage_summary()
print(f"Filled {summary['bins_filled']}/{sampler.num_bins} bins")
print(f"Used {summary['total_evaluations']} evaluations")
```

## Conclusion

The binary search approach is well-suited for sampling monotonic curves. It efficiently ensures uniform coverage along the AFHP axis while respecting the constraint that we can only sample points on the actual performance curve. This makes it an effective solution for threshold evaluation in coordination policies.
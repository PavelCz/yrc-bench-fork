# OOD Score Not Showing in Videos - Analysis and Fix

## Problem
When running `eval_afhp.py` with maze configuration (which uses `use_random_env_switch: True`), OOD scores are not being saved to videos, even though they work correctly with coinrun configuration.

## Root Cause Analysis

The issue was caused by **missing configuration parameters** and **lack of defensive coding** in the `ThresholdPolicy`:

1. **Missing Config Parameters**: Both `coinrun_bg_threshold.yaml` and `maze/threshold.yaml` were missing the `rolling_average` and `rolling_average_size` parameters in the `coord_policy` section.

2. **Unsafe Attribute Access**: The `ThresholdPolicy.__init__()` method was directly accessing `self.args.rolling_average` and `self.args.rolling_average_size` without defaults, which could cause `AttributeError` if these attributes were not set.

3. **Silent Failures**: If an exception occurred in `ThresholdPolicy.act()`, it would propagate up and potentially cause scores to be `None`.

4. **Observation/Action Space Mismatches**: When using `RandomEnvSwitchWrapper` to switch between `maze_aisc` and `maze` environments, there was no validation that these environments had compatible observation and action spaces.

## Fixes Applied

### 1. Made ThresholdPolicy More Robust
**File**: `YRC/policies/threshold.py`

Changed unsafe attribute access to use `getattr()` with defaults:
```python
# Before:
self.rolling_average: Optional[str] = self.args.rolling_average
self.rolling_average_size: int = self.args.rolling_average_size

# After:
self.rolling_average: Optional[str] = getattr(self.args, "rolling_average", None)
self.rolling_average_size: int = getattr(self.args, "rolling_average_size", 10)
```

Added error handling in `act()` method to catch exceptions and return safe defaults instead of propagating errors.

### 2. Added Missing Config Parameters
**Files**: 
- `configs/eval/coinrun_bg_threshold.yaml`
- `configs/eval/maze/threshold.yaml`

Added explicit values for:
```yaml
coord_policy:
    cls: ThresholdPolicy
    feature_type: 'obs'
    collect_data_agent: 'weak'
    metric: 'max_prob'
    rolling_average: 'none'  # NEW
    rolling_average_size: 5   # NEW
```

### 3. Added Debug Logging
**File**: `YRC/core/evaluator.py`

Added logging to track:
- First scores collected from policy
- Scores returned by policy on each call
- Episodes with all None scores (with warnings)
- Count of non-None scores per episode

### 4. Added Environment Compatibility Checks
**File**: `YRC/envs/procgen/wrappers.py`

Added validation in `RandomEnvSwitchWrapper.__init__()` to warn if:
- Observation spaces have different shapes
- Action spaces have different numbers of actions

These mismatches can cause issues with policies that expect specific shapes.

## Testing

To test if the fix works:

1. Run eval_afhp with coinrun:
```bash
python eval_afhp.py -config configs/eval/coinrun_bg_threshold.yaml
```

2. Run eval_afhp with maze:
```bash
python eval_afhp.py -config configs/eval/maze/threshold.yaml
```

3. Check the log output for:
   - Any warnings about observation/action space mismatches
   - Debug messages about scores being collected
   - Warning messages about episodes with all None scores

4. Check the generated videos to verify that OOD score bars are visible at the top of the frames.

## Additional Notes

- The `RandomEnvSwitchWrapper` switches between two environments (`maze_aisc` and `maze` in the maze config). If these environments have different observation or action spaces, the weak agent (trained on one environment) may not work properly on the other.

- Consider using environments with identical observation and action spaces when using `RandomEnvSwitchWrapper`.

- The debug logging can be controlled by setting the logging level. For more verbose output, add `--log-level DEBUG` to the command line.

## Related Files
- `YRC/policies/threshold.py` - Policy implementation
- `YRC/core/evaluator.py` - Evaluation loop and video collection
- `YRC/core/video_utils.py` - Video generation with score bars
- `YRC/envs/procgen/wrappers.py` - Environment wrappers including RandomEnvSwitchWrapper
- `configs/eval/coinrun_bg_threshold.yaml` - Coinrun config
- `configs/eval/maze/threshold.yaml` - Maze config

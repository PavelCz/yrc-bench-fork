# Implementation Plan: Remove RandomEnvSwitchWrapper and Add random_percent for Maze

## Overview

This document outlines a plan to:
1. Remove all `RandomEnvSwitchWrapper` functionality from the codebase
2. Create a new `maze_afh` environment with `random_percent` support

The goal is to unify the mechanism for creating "mixed" environments (ID/OOD splits) across different Procgen games. Currently, coinrun uses `random_percent` to probabilistically switch between deterministic and random coin placement, while maze relies on the external `RandomEnvSwitchWrapper` to switch between `maze` and `maze_aisc` environments. After this change, `maze_afh` will use the same `random_percent` pattern as coinrun, while preserving the original `maze` and `maze_aisc` environments for backward compatibility.

## Background

### Current random_percent Implementation (coinrun)

In `coinrun.cpp`, the `random_percent` option controls whether the coin is placed at a fixed location (end of level) or randomly:

```cpp
// In game_reset():
int rand_check = rand_gen.randn(100);
randomize_goal = (rand_check < options.random_percent);
generate_coin(randomize_goal);
```

The value flows through:
1. Python: `ProcgenGym3Env.__init__` receives `random_percent` parameter
2. Python: Passed to C++ via `options["random_percent"]`
3. C++: `game.cpp` consumes it into `options.random_percent`
4. C++: Game-specific code uses it in `game_reset()`

### Current Maze vs Maze_AISC Difference

- **maze.cpp**: Uses `maze_gen->place_objects(GOAL, 1)` - places cheese randomly anywhere in the maze
- **maze_aisc.cpp**: Uses `maze_gen->deterministic_place(GOAL, false, rand_region)` - places cheese in top-right corner (or within a region if `rand_region > 0`)

### Current RandomEnvSwitchWrapper Usage

Files that use or reference `RandomEnvSwitchWrapper`:

| File | Usage |
|------|-------|
| `lib/procgen/procgen/wrappers.py` | Primary wrapper implementation |
| `lib/procgen/procgen/__init__.py` | Exports `RandomEnvSwitchWrapper` |
| `YRC/envs/procgen/wrappers.py` | Duplicate wrapper implementation |
| `lib/train-procgen-pytorch/train.py` | Uses for training with `--switch_env_names` |
| `eval_afhp.py` | Uses for evaluation with `use_random_env_switch` config |
| `YRC/core/evaluator.py` | Handles `random_env_switch` flag for OOD ground-truth |
| `lib/train-procgen-pytorch/RANDOM_ENV_SWITCH_USAGE.md` | Documentation |
| `configs/eval/maze/timestep_random.yaml` | Config using random_env_switch |
| `configs/eval/maze/threshold.yaml` | Config using random_env_switch |
| `configs/eval/maze/level_based_random.yaml` | Config using random_env_switch |
| `configs/eval/maze/timestep_random_maze_hard.yaml` | Config using random_env_switch |
| `configs/eval/heist/timestep_random.yaml` | Config using random_env_switch |
| `configs/eval/heist/threshold.yaml` | Config using random_env_switch |
| `configs/eval/heist/timestep_random_many_chests.yaml` | Config using random_env_switch |
| `configs/eval/heist/timestep_random_many_keys.yaml` | Config using random_env_switch |
| `configs/eval/random_env_switch_example.yaml` | Example config |

---

## Part 1: Create maze_afh with random_percent Support

Instead of modifying the existing `maze.cpp`, we'll create a new environment variant called `maze_afh` ("ask for help") that supports `random_percent`. This preserves backward compatibility with existing experiments using `maze` and `maze_aisc`.

### 1.1 Create maze_afh.cpp

**New File:** `lib/procgen/procgen/src/games/maze_afh.cpp`

Copy `maze.cpp` and make the following changes:

1. Update the name constant:
   ```cpp
   const std::string NAME = "maze_afh";
   ```

2. Rename the class:
   ```cpp
   class MazeGameAFH : public BasicAbstractGame {
   ```

3. Add member variables to track goal placement (similar to coinrun):
   ```cpp
   bool randomize_goal = false;  // whether goal was randomly placed
   bool prev_level_randomize_goal = false;
   ```

4. Modify `game_reset()` to use `random_percent`:
   ```cpp
   void game_reset() override {
       BasicAbstractGame::game_reset();
       // ... existing maze setup code ...
       
       maze_gen->generate_maze();
       
       // Decide goal placement based on random_percent
       // random_percent=0 means always deterministic (corner)
       // random_percent=100 means always random placement
       int rand_check = rand_gen.randn(100);
       prev_level_randomize_goal = randomize_goal;
       randomize_goal = (rand_check < options.random_percent);
       
       if (randomize_goal) {
           // Random placement (original maze behavior)
           maze_gen->place_objects(GOAL, 1);
       } else {
           // Deterministic placement in top-right corner (maze_aisc behavior)
           // Use rand_region=0 for exact corner placement
           maze_gen->deterministic_place(GOAL, false, options.rand_region);
       }
       
       // ... rest of existing grid copy code ...
   }
   ```

5. Add info observation for tracking:
   ```cpp
   void observe() override {
       Game::observe();
       *(int32_t *)(info_bufs[info_name_to_offset.at("randomize_goal")]) = randomize_goal;
       *(int32_t *)(info_bufs[info_name_to_offset.at("prev_level/randomize_goal")]) = prev_level_randomize_goal;
   }
   ```

6. Update serialization (for state save/load):
   ```cpp
   void serialize(WriteBuffer *b) override {
       BasicAbstractGame::serialize(b);
       b->write_int(maze_dim);
       b->write_int(world_dim);
       b->write_bool(randomize_goal);
   }

   void deserialize(ReadBuffer *b) override {
       BasicAbstractGame::deserialize(b);
       maze_dim = b->read_int();
       world_dim = b->read_int();
       randomize_goal = b->read_bool();
   }
   ```

7. Update the registration macro:
   ```cpp
   REGISTER_GAME(NAME, MazeGameAFH);
   ```

### 1.2 Register maze_afh in env.py

**File:** `lib/procgen/procgen/env.py`

Add `"maze_afh"` to the `ENV_NAMES` list:
```python
ENV_NAMES = [
    # ... existing envs ...
    "maze",
    "maze_fixed_size",
    "maze_aisc",
    "maze_afh",  # NEW: maze with random_percent support
    # ... rest of envs ...
]
```

Add to `EXPLORATION_LEVEL_SEEDS` if needed:
```python
EXPLORATION_LEVEL_SEEDS = {
    # ... existing ...
    "maze_afh": 158988835,  # Same as maze
    # ...
}
```

### 1.3 Update vecgame.cpp for info buffers (if needed)

**File:** `lib/procgen/procgen/src/vecgame.cpp`

If `maze_afh` needs new info keys (`randomize_goal`, `prev_level/randomize_goal`), add them to the info buffer setup. Check how coinrun registers these - the info keys may already be available globally.

### 1.4 Testing maze_afh

After implementation:
```bash
# Test with 0% random (all deterministic, cheese in corner)
python -c "from procgen import ProcgenEnv; env = ProcgenEnv(num_envs=1, env_name='maze_afh', random_percent=0); ..."

# Test with 100% random (all random placement)
python -c "from procgen import ProcgenEnv; env = ProcgenEnv(num_envs=1, env_name='maze_afh', random_percent=100); ..."

# Test with 50% mix
python -c "from procgen import ProcgenEnv; env = ProcgenEnv(num_envs=1, env_name='maze_afh', random_percent=50); ..."
```

### 1.5 Behavior Summary

| `random_percent` | Cheese Placement |
|------------------|------------------|
| 0 | Always in top-right corner (like `maze_aisc`) |
| 100 | Always random (like original `maze`) |
| 50 | 50% corner, 50% random |

This allows experiments to use a single environment with probabilistic OOD switching, matching how `coinrun` works with `random_percent`.

---

## Part 2: Remove RandomEnvSwitchWrapper

### 2.1 Remove from lib/procgen

**Files to modify:**

1. **`lib/procgen/procgen/wrappers.py`**
   - Delete the entire `RandomEnvSwitchWrapper` class

2. **`lib/procgen/procgen/__init__.py`**
   - Remove the import/export of `RandomEnvSwitchWrapper`

### 2.2 Remove from YRC codebase

**Files to modify:**

1. **`YRC/envs/procgen/wrappers.py`**
   - Delete the `RandomEnvSwitchWrapper` class (lines ~428-566)

### 2.3 Remove from train-procgen-pytorch

**Files to modify:**

1. **`lib/train-procgen-pytorch/train.py`**
   - Remove import: `from procgen import ProcgenEnv, RandomEnvSwitchWrapper` → `from procgen import ProcgenEnv`
   - Remove argument definitions:
     - `--switch_env_names`
     - `--switch_percent`
     - `--switch_env_names_eval`
     - `--switch_percent_eval`
   - Remove the `RandomEnvSwitchWrapper` logic in `create_venv()` function (lines ~169-217)

2. **`lib/train-procgen-pytorch/RANDOM_ENV_SWITCH_USAGE.md`**
   - Delete this file entirely

### 2.4 Remove from eval_afhp.py

**File:** `eval_afhp.py`

**Changes:**
- Remove import: `from procgen import RandomEnvSwitchWrapper`
- Remove the `use_random_env_switch` logic block (~lines 133-192)
- Remove the `create_raw_env_from_config()` function
- Simplify the `main()` function to always use `env_factory.make()` without random env switch handling

### 2.5 Remove from YRC/core/evaluator.py

**File:** `YRC/core/evaluator.py`

**Changes:**
- Remove `random_env_switch` parameter from `__init__`
- Remove `self.random_env_switch` attribute
- Remove `_random_env_switch_is_ood()` method
- Remove all `if self.random_env_switch:` conditional blocks in:
  - `eval()` method
  - `_eval_loop()` method
- Update OOD ground truth determination to always use `info["randomize_goal"]`

### 2.6 Update config files

**Files to update:**

Replace `use_random_env_switch` configs with `maze_afh` and `random_percent`:

1. **`configs/eval/maze/timestep_random.yaml`**
   - Remove `use_random_env_switch: True` and `random_env_switch:` section
   - Change `env_name: 'maze'` to `env_name: 'maze_afh'`
   - Set `random_percent: 50` in the environment configs

2. **`configs/eval/maze/threshold.yaml`** - Same changes

3. **`configs/eval/maze/level_based_random.yaml`** - Same changes

4. **`configs/eval/maze/timestep_random_maze_hard.yaml`** - Same changes

5. **`configs/eval/heist/`** configs - These will need a separate `heist_afh` implementation or alternative approach

6. **`configs/eval/random_env_switch_example.yaml`**
   - Delete this file or convert to a `maze_afh` + `random_percent` example

### 2.7 Update README.md

**File:** `README.md`

Update any documentation that references `RandomEnvSwitchWrapper` to describe the new `random_percent` approach for maze.

---

## Part 3: Verification Checklist

### After Implementation

- [ ] `maze_afh` environment compiles and is registered
- [ ] `maze_afh` with `random_percent=0` places cheese in top-right corner
- [ ] `maze_afh` with `random_percent=100` places cheese randomly
- [ ] `maze_afh` with `random_percent=50` produces ~50/50 split
- [ ] `info["randomize_goal"]` correctly reports placement type for `maze_afh`
- [ ] Original `maze` and `maze_aisc` environments still work unchanged
- [ ] Training with `lib/train-procgen-pytorch/train.py` works without `--switch_env_names`
- [ ] Evaluation with `eval_afhp.py` works without `use_random_env_switch`
- [ ] All maze config files updated to use `maze_afh` and working
- [ ] No remaining references to `RandomEnvSwitchWrapper` in codebase

### Grep verification

After all changes, these should return no results:
```bash
grep -r "RandomEnvSwitchWrapper" --include="*.py" --include="*.cpp" --include="*.yaml"
grep -r "switch_env_names" --include="*.py"
grep -r "use_random_env_switch" --include="*.yaml"
```

---

## Part 4: Migration Guide for Existing Experiments

### Before (RandomEnvSwitchWrapper)

```yaml
# Config file
evaluation:
    use_random_env_switch: True
    random_env_switch:
        env1:
            gym_name: 'maze_aisc'
        env2:
            gym_name: 'maze'
        random_percent: 50
```

```bash
# Training command
python train.py --switch_env_names maze maze_aisc --switch_percent 50
```

### After (random_percent with maze_afh)

```yaml
# Config file
environment:
    common:
        env_name: 'maze_afh'  # Use new maze_afh environment
    test:
        random_percent: 50  # 50% random placement, 50% corner placement
```

```bash
# Training command
python train.py --env_name maze_afh --random_percent 50
```

---

## Part 5: Future Considerations

### Extending to Other Environments

The `random_percent` pattern could be extended to other environments that need ID/OOD splits:

- **heist**: Could use `random_percent` to control key/chest placement (would require creating `heist_afh.cpp`)
- **coinrun**: Already supports `random_percent`

### Relationship Between Maze Variants

After this change, we will have:

| Environment | Cheese Placement | `random_percent` Support |
|-------------|------------------|--------------------------|
| `maze` | Always random | No (ignored) |
| `maze_aisc` | Always corner (or region via `rand_region`) | No (ignored) |
| `maze_afh` | Probabilistic based on `random_percent` | **Yes** |

- `maze_afh` with `random_percent=0` ≈ `maze_aisc` with `rand_region=0`
- `maze_afh` with `random_percent=100` = `maze` behavior
- `maze_afh` with `random_percent=50` = 50/50 split (replaces `RandomEnvSwitchWrapper` use case)

The original `maze` and `maze_aisc` environments are preserved for backward compatibility with existing experiments.

---

## Implementation Order

Recommended implementation order:

1. **Phase 1: Create maze_afh environment** (Part 1)
   - Copy `maze.cpp` to `maze_afh.cpp`
   - Add `random_percent` support
   - Register in `env.py`
   - This is additive and doesn't break anything
   - Test thoroughly before proceeding

2. **Phase 2: Update configs** (Part 2.6)
   - Update maze configs to use `maze_afh` with `random_percent`
   - Verify experiments still work

3. **Phase 3: Remove RandomEnvSwitchWrapper** (Parts 2.1-2.5)
   - Remove wrapper code
   - Update train.py and eval_afhp.py
   - Remove from evaluator.py

4. **Phase 4: Cleanup** (Part 2.7)
   - Update documentation
   - Run verification checklist

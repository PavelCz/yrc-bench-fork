# Implementation Plan: Create heist_afh with random_percent Support

## Overview

This document outlines a plan to create a new `heist_afh` environment with `random_percent` support, similar to how `maze_afh` was implemented. The `random_percent` parameter will control the probabilistic switching between:
- **ID (In-Distribution)**: `heist_aisc_many_chests` behavior (more chests than keys)
- **OOD (Out-of-Distribution)**: `heist_aisc_many_keys` behavior (more keys than chests)

This unifies the mechanism for creating "mixed" environments across different Procgen games, replacing the need for `RandomEnvSwitchWrapper` for heist.

## Background

### Current Heist Variants

| Environment | Keys:Chests Ratio | Description |
|-------------|-------------------|-------------|
| `heist_aisc_many_chests` | 1:2 | `num_keys` is base, `env_chests = num_keys * 2` |
| `heist_aisc_many_keys` | 2:1 | `env_chests` is base, `num_keys = env_chests * 2` |

### Key Differences in game_reset()

**heist_aisc_many_chests.cpp** (lines 189-197):
```cpp
// MANY CHESTS SETTING
num_keys = difficulty + rand_gen.randn(2) + 1;
env_chests = num_keys * 2;
```

**heist_aisc_many_keys.cpp** (lines 188-194):
```cpp
env_chests = difficulty + rand_gen.randn(2) + 1;
num_keys = env_chests * 2;
```

### Reference: maze_afh Implementation Pattern

From `maze_afh.cpp`:
```cpp
// In game_reset():
prev_level_randomize_goal = randomize_goal;
int rand_check = rand_gen.randn(100);
randomize_goal = (rand_check < options.random_percent);

if (randomize_goal) {
    // Random/OOD placement
    maze_gen->place_objects(GOAL, 1);
} else {
    // Deterministic/ID placement
    maze_gen->deterministic_place(GOAL, false, rand_region);
}
```

### Current RandomEnvSwitchWrapper Usage for Heist

Files that use `RandomEnvSwitchWrapper` with heist:

| File | Usage |
|------|-------|
| `configs/eval/heist/timestep_random.yaml` | Switches between `heist_aisc_many_chests` and `heist_aisc_many_keys` |
| `configs/eval/heist/threshold.yaml` | Same |
| `configs/eval/heist/timestep_random_many_chests.yaml` | Same |
| `configs/eval/heist/timestep_random_many_keys.yaml` | Same |

---

## Part 1: Create heist_afh.cpp

### 1.1 Create the New File

**New File:** `lib/procgen/procgen/src/games/heist_afh.cpp`

Copy `heist_aisc_many_chests.cpp` and make the following changes:

### 1.2 Update Name and Class

```cpp
const std::string NAME = "heist_afh";

class HeistGameAFH : public BasicAbstractGame {
```

### 1.3 Add Member Variables for OOD Tracking

Add after existing member variables:
```cpp
// Track object placement mode for OOD detection (similar to maze_afh/coinrun)
bool randomize_placement = false;  // whether OOD (many_keys) placement was used this level
bool prev_level_randomize_placement = false;  // placement mode from previous level
```

### 1.4 Modify game_reset() for random_percent Support

Replace the key/chest calculation logic:

```cpp
void game_reset() override {
    BasicAbstractGame::game_reset();

    int min_maze_dim = 5;
    int max_diff = (world_dim - min_maze_dim) / 2;
    int difficulty = rand_gen.randn(max_diff + 1);
    options.center_agent = options.distribution_mode == MemoryMode;

    // Decide placement mode based on random_percent (like maze_afh/coinrun)
    // random_percent=0 means always ID (many_chests behavior)
    // random_percent=100 means always OOD (many_keys behavior)
    prev_level_randomize_placement = randomize_placement;
    int rand_check = rand_gen.randn(100);
    randomize_placement = (rand_check < options.random_percent);

    if (options.distribution_mode == MemoryMode) {
        if (randomize_placement) {
            // OOD: many_keys behavior (more keys than chests)
            env_chests = rand_gen.randn(4) + 1;
            num_keys = env_chests * 2;
        } else {
            // ID: many_chests behavior (more chests than keys)
            num_keys = rand_gen.randn(4) + 1;
            env_chests = num_keys * 2;
        }
    } else {
        if (randomize_placement) {
            // OOD: many_keys behavior (more keys than chests)
            env_chests = difficulty + rand_gen.randn(2) + 1;
            num_keys = env_chests * 2;
        } else {
            // ID: many_chests behavior (more chests than keys)
            num_keys = difficulty + rand_gen.randn(2) + 1;
            env_chests = num_keys * 2;
        }
    }

    agent_keys = 0;
    total_chests = env_chests;

    // ... rest of game_reset() unchanged ...
}
```

### 1.5 Add observe() Method for Info Exposure

Add after `game_step()`:
```cpp
// Expose placement mode info for OOD detection (like maze_afh/coinrun)
void observe() override {
    Game::observe();
    *(int32_t *)(info_bufs[info_name_to_offset.at("randomize_goal")]) = randomize_placement;
    *(int32_t *)(info_bufs[info_name_to_offset.at("prev_level/randomize_goal")]) = prev_level_randomize_placement;
}
```

Note: We use `"randomize_goal"` as the key name for consistency with `maze_afh` and `coinrun`, even though for heist it represents the key/chest ratio rather than goal placement.

### 1.6 Update Serialization

```cpp
void serialize(WriteBuffer *b) override {
    BasicAbstractGame::serialize(b);
    b->write_int(num_keys);
    b->write_int(world_dim);
    b->write_vector_bool(has_keys);
    b->write_bool(randomize_placement);
}

void deserialize(ReadBuffer *b) override {
    BasicAbstractGame::deserialize(b);
    num_keys = b->read_int();
    world_dim = b->read_int();
    has_keys = b->read_vector_bool();
    randomize_placement = b->read_bool();
}
```

### 1.7 Update Registration Macro

```cpp
REGISTER_GAME(NAME, HeistGameAFH);
```

---

## Part 2: Register heist_afh in env.py

### 2.1 Add to ENV_NAMES

**File:** `lib/procgen/procgen/env.py`

Add `"heist_afh"` to the `ENV_NAMES` list:
```python
ENV_NAMES = [
    # ... existing envs ...
    "heist",
    "heist_aisc_many_chests",
    "heist_aisc_many_keys",
    "heist_afh",  # NEW: heist with random_percent support
    # ... rest of envs ...
]
```

### 2.2 Add to EXPLORATION_LEVEL_SEEDS (if needed)

```python
EXPLORATION_LEVEL_SEEDS = {
    # ... existing ...
    "heist_afh": ...,  # Copy from heist or heist_aisc_many_chests
    # ...
}
```

---

## Part 3: Testing heist_afh

### 3.1 Basic Functionality Tests

```bash
# Test with 0% random (all ID - many_chests behavior)
python -c "
from procgen import ProcgenEnv
env = ProcgenEnv(num_envs=1, env_name='heist_afh', random_percent=0, distribution_mode='hard')
obs = env.reset()
info = env.callmethod('get_info')
print('random_percent=0, randomize_goal:', info[0].get('randomize_goal', 'N/A'))
"

# Test with 100% random (all OOD - many_keys behavior)
python -c "
from procgen import ProcgenEnv
env = ProcgenEnv(num_envs=1, env_name='heist_afh', random_percent=100, distribution_mode='hard')
obs = env.reset()
info = env.callmethod('get_info')
print('random_percent=100, randomize_goal:', info[0].get('randomize_goal', 'N/A'))
"

# Test with 50% mix
python -c "
from procgen import ProcgenEnv
env = ProcgenEnv(num_envs=100, env_name='heist_afh', random_percent=50, distribution_mode='hard')
obs = env.reset()
info = env.callmethod('get_info')
randomize_count = sum(1 for i in info if i.get('randomize_goal', 0))
print(f'random_percent=50, OOD count: {randomize_count}/100 (~50% expected)')
"
```

### 3.2 Verify Behavior Matches Original Environments

```bash
# Verify heist_afh with random_percent=0 matches heist_aisc_many_chests
# Verify heist_afh with random_percent=100 matches heist_aisc_many_keys
# (Compare key/chest counts in generated levels)
```

---

## Part 4: Update Config Files

### 4.1 Convert Heist Configs to Use heist_afh

**Before (RandomEnvSwitchWrapper):**
```yaml
evaluation:
    use_random_env_switch: True
    random_env_switch:
        env1:
            gym_name: 'heist_aisc_many_chests'
        env2:
            gym_name: 'heist_aisc_many_keys'
        random_percent: 50
```

**After (heist_afh with random_percent):**
```yaml
environment:
    common:
        env_name: 'heist_afh'
    train:
        random_percent: 0  # ID only for training
    val_sim:
        random_percent: 50
    val_true:
        random_percent: 50
    test:
        random_percent: 50
```

### 4.2 Files to Update

1. `configs/eval/heist/timestep_random.yaml`
2. `configs/eval/heist/threshold.yaml`
3. `configs/eval/heist/timestep_random_many_chests.yaml`
4. `configs/eval/heist/timestep_random_many_keys.yaml`

---

## Part 5: Behavior Summary

| `random_percent` | Behavior | Keys:Chests |
|------------------|----------|-------------|
| 0 | Always ID (like `heist_aisc_many_chests`) | 1:2 |
| 100 | Always OOD (like `heist_aisc_many_keys`) | 2:1 |
| 50 | 50% ID, 50% OOD | Mixed |

### Info Observation

| Key | Value | Description |
|-----|-------|-------------|
| `randomize_goal` | 0 or 1 | Whether current level uses OOD (many_keys) placement |
| `prev_level/randomize_goal` | 0 or 1 | Placement mode from previous level |

---

## Part 6: Verification Checklist

### After Implementation

- [ ] `heist_afh` environment compiles and is registered
- [ ] `heist_afh` with `random_percent=0` uses many_chests behavior (1:2 keys:chests)
- [ ] `heist_afh` with `random_percent=100` uses many_keys behavior (2:1 keys:chests)
- [ ] `heist_afh` with `random_percent=50` produces ~50/50 split
- [ ] `info["randomize_goal"]` correctly reports placement mode for `heist_afh`
- [ ] Original `heist_aisc_many_chests` and `heist_aisc_many_keys` environments still work unchanged
- [ ] All heist config files updated to use `heist_afh` and working
- [ ] Serialization/deserialization works correctly

### Grep Verification (after RandomEnvSwitchWrapper removal)

```bash
grep -r "heist_aisc_many" configs/eval/heist/ --include="*.yaml"
# Should only find references in backward-compatibility comments or none
```

---

## Part 7: Complete heist_afh.cpp Template

```cpp
#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include <set>
#include <queue>
#include "../mazegen.h"
#include "../cpp-utils.h"

const std::string NAME = "heist_afh";

const float COMPLETION_BONUS = 10.0f;

const int LOCKED_DOOR = 1;
const int KEY = 2;
const int EXIT = 9;
const int KEY_ON_RING = 11;

class HeistGameAFH : public BasicAbstractGame {
  public:
    std::shared_ptr<MazeGen> maze_gen_aisc;
    int world_dim = 0;
    int num_keys = 0;
    std::vector<bool> has_keys;
    int agent_keys = 0;
    int env_chests = 0;
    int total_chests = 0;

    // Track placement mode for OOD detection (similar to maze_afh/coinrun)
    bool randomize_placement = false;  // whether OOD (many_keys) placement was used
    bool prev_level_randomize_placement = false;  // placement mode from previous level

    HeistGameAFH()
        : BasicAbstractGame(NAME) {
        maze_gen_aisc = nullptr;
        has_useful_vel_info = false;

        main_width = 20;
        main_height = 20;

        out_of_bounds_object = WALL_OBJ;
        visibility = 8.0;
    }

    void load_background_images() override {
        main_bg_images_ptr = &topdown_backgrounds;
    }

    bool should_preserve_type_themes(int type) override {
        return type == KEY || type == LOCKED_DOOR;
    }

    void asset_for_type(int type, std::vector<std::string> &names) override {
        if (type == WALL_OBJ) {
            names.push_back("kenney/Ground/Dirt/dirtCenter.png");
        } else if (type == EXIT) {
            names.push_back("misc_assets/gemYellow.png");
        } else if (type == PLAYER) {
            names.push_back("misc_assets/spaceAstronauts_008.png");
        } else if (type == KEY) {
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyGreen.png");
        } else if (type == LOCKED_DOOR) {
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
            names.push_back("misc_assets/lock_blue.png");
        }
    }

    bool use_block_asset(int type) override {
        return BasicAbstractGame::use_block_asset(type) || (type == WALL_OBJ);
    }

    bool is_blocked_ents(const std::shared_ptr<Entity> &src, const std::shared_ptr<Entity> &target, bool is_horizontal) override {
        if (target->type == LOCKED_DOOR)
            return false;

        return BasicAbstractGame::is_blocked_ents(src, target, is_horizontal);
    }

    bool should_draw_entity(const std::shared_ptr<Entity> &entity) override {
        if (entity->type == KEY_ON_RING)
            return entity->image_theme < agent_keys;

        return BasicAbstractGame::should_draw_entity(entity);
    }

    void handle_agent_collision(const std::shared_ptr<Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        if (obj->type == KEY) {
            obj->will_erase = true;
            step_data.reward -= options.key_penalty / 10.;
            agent_keys += 1;
        } else if (obj->type == LOCKED_DOOR) {
            if (agent_keys > 0) {
                obj->will_erase = true;
                agent_keys += -1;
                env_chests += -1;
                step_data.reward = 1;
            }
            if (env_chests == 0) {
                step_data.done = true;
                step_data.level_complete = true;
            }
            if (total_chests - num_keys == env_chests) {
                step_data.done = true;
                step_data.level_complete = true;
            }
        }
    }

    void choose_world_dim() override {
        int dist_diff = options.distribution_mode;

        if (dist_diff == EasyMode) {
            world_dim = 9;
        } else if (dist_diff == HardMode) {
            world_dim = 13;
        } else if (dist_diff == MemoryMode) {
            world_dim = 23;
        }

        maxspeed = .75;

        main_width = world_dim;
        main_height = world_dim;
    }

    void game_reset() override {
        BasicAbstractGame::game_reset();

        int min_maze_dim = 5;
        int max_diff = (world_dim - min_maze_dim) / 2;
        int difficulty = rand_gen.randn(max_diff + 1);
        options.center_agent = options.distribution_mode == MemoryMode;

        // Decide placement mode based on random_percent (like maze_afh/coinrun)
        // random_percent=0 means always ID (many_chests behavior)
        // random_percent=100 means always OOD (many_keys behavior)
        prev_level_randomize_placement = randomize_placement;
        int rand_check = rand_gen.randn(100);
        randomize_placement = (rand_check < options.random_percent);

        if (options.distribution_mode == MemoryMode) {
            if (randomize_placement) {
                // OOD: many_keys behavior (more keys than chests)
                env_chests = rand_gen.randn(4) + 1;
                num_keys = env_chests * 2;
            } else {
                // ID: many_chests behavior (more chests than keys)
                num_keys = rand_gen.randn(4) + 1;
                env_chests = num_keys * 2;
            }
        } else {
            if (randomize_placement) {
                // OOD: many_keys behavior (more keys than chests)
                env_chests = difficulty + rand_gen.randn(2) + 1;
                num_keys = env_chests * 2;
            } else {
                // ID: many_chests behavior (more chests than keys)
                num_keys = difficulty + rand_gen.randn(2) + 1;
                env_chests = num_keys * 2;
            }
        }

        agent_keys = 0;
        total_chests = env_chests;

        has_keys.clear();

        for (int i = 0; i < num_keys; i++) {
            has_keys.push_back(false);
        }

        int maze_dim = difficulty * 2 + min_maze_dim;
        float maze_scale = main_height / (world_dim * 1.0);

        agent->rx = .375 * maze_scale;
        agent->ry = .375 * maze_scale;

        float r_ent = maze_scale / 2;

        maze_gen_aisc = std::make_shared<MazeGen>(&rand_gen, maze_dim);
        maze_gen_aisc->generate_maze_with_doors_aisc(env_chests, num_keys);

        // move agent out of the way for maze generation
        agent->x = -1;
        agent->y = -1;

        int off_x = rand_gen.randn(world_dim - maze_dim + 1);
        int off_y = rand_gen.randn(world_dim - maze_dim + 1);

        for (int i = 0; i < grid_size; i++) {
            set_obj(i, WALL_OBJ);
        }

        for (int i = 0; i < maze_dim; i++) {
            for (int j = 0; j < maze_dim; j++) {
                int x = off_x + i;
                int y = off_y + j;

                int obj = maze_gen_aisc->grid.get(i + MAZE_OFFSET, j + MAZE_OFFSET);

                float obj_x = (x + .5) * maze_scale;
                float obj_y = (y + .5) * maze_scale;

                if (obj != WALL_OBJ) {
                    set_obj(x, y, SPACE);
                }
                if (obj >= KEY_OBJ) {
                    auto ent = spawn_entity(.375 * maze_scale, KEY, maze_scale * x, maze_scale * y, maze_scale, maze_scale);
                    ent->image_theme = obj - KEY_OBJ - 1;
                    match_aspect_ratio(ent);
                } else if (obj >= DOOR_OBJ) {
                    auto ent = add_entity(obj_x, obj_y, 0, 0, r_ent, LOCKED_DOOR);
                    ent->image_theme = obj - DOOR_OBJ - 1;
                } else if (obj == EXIT_OBJ) {
                    auto ent = spawn_entity(.375 * maze_scale, EXIT, maze_scale * x, maze_scale * y, maze_scale, maze_scale);
                    match_aspect_ratio(ent);
                } else if (obj == AGENT_OBJ) {
                    agent->x = obj_x;
                    agent->y = obj_y;
                }
            }
        }

        float ring_key_r = 0.03f;

        for (int i = 0; i < num_keys; i++) {
            auto ent = add_entity(1 - ring_key_r * (2 * i + 1.25), ring_key_r * .75, 0, 0, ring_key_r, KEY_ON_RING);
            ent->image_theme = i;
            ent->image_type = KEY;
            ent->rotation = PI / 2;
            ent->render_z = 1;
            ent->use_abs_coords = true;
            match_aspect_ratio(ent);
        }
    }

    void game_step() override {
        BasicAbstractGame::game_step();

        step_data.reward -= options.step_penalty / 1000.;
        agent->face_direction(action_vx, action_vy);
    }

    void serialize(WriteBuffer *b) override {
        BasicAbstractGame::serialize(b);
        b->write_int(num_keys);
        b->write_int(world_dim);
        b->write_vector_bool(has_keys);
        b->write_bool(randomize_placement);
    }

    void deserialize(ReadBuffer *b) override {
        BasicAbstractGame::deserialize(b);
        num_keys = b->read_int();
        world_dim = b->read_int();
        has_keys = b->read_vector_bool();
        randomize_placement = b->read_bool();
    }

    // Expose placement mode info for OOD detection (like maze_afh/coinrun)
    void observe() override {
        Game::observe();
        *(int32_t *)(info_bufs[info_name_to_offset.at("randomize_goal")]) = randomize_placement;
        *(int32_t *)(info_bufs[info_name_to_offset.at("prev_level/randomize_goal")]) = prev_level_randomize_placement;
    }
};

REGISTER_GAME(NAME, HeistGameAFH);
```

---

## Implementation Order

1. **Phase 1: Create heist_afh.cpp**
   - Copy from `heist_aisc_many_chests.cpp`
   - Add `random_percent` logic
   - Add `observe()` method
   - Update serialization
   - This is additive and doesn't break anything

2. **Phase 2: Register in env.py**
   - Add `"heist_afh"` to `ENV_NAMES`

3. **Phase 3: Test**
   - Verify basic functionality
   - Verify behavior matches original environments at extremes

4. **Phase 4: Update configs**
   - Update heist configs to use `heist_afh` with `random_percent`

5. **Phase 5: Remove RandomEnvSwitchWrapper usage** (if desired)
   - Remove `use_random_env_switch` from heist configs
   - This can be done after maze configs are also updated

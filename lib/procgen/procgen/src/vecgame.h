#pragma once

/*

This implements the libenv interface and manages a vector of Game instances

*/

#include <memory>
#include <vector>
#include <mutex>
#include <string>
#include <condition_variable>
#include <thread>
#include <list>
#include <atomic>
#include <random>

class VecOptions;
class Game;

class VecGame {
  public:
    std::vector<struct libenv_tensortype> observation_types;
    std::vector<struct libenv_tensortype> action_types;
    std::vector<struct libenv_tensortype> info_types;

    int num_envs;
    int num_joint_games;
    int num_actions;
    bool render_human;

    std::vector<std::shared_ptr<Game>> games;

    // Shared seed list for ordered level seeds across all envs
    std::vector<int> shared_level_seeds;
    std::atomic<int> shared_seed_index{0};
    std::mutex seed_mutex;
    
    // Container mode: random draw with refill when empty
    bool seed_container_mode = false;
    std::vector<int> available_seeds;  // seeds remaining in current container cycle
    std::mt19937 container_rng;        // RNG for container mode random selection
    
    // Random mode: sample with replacement (always pick randomly from full list)
    bool seed_random_mode = false;

    VecGame(int _nenvs, VecOptions opt_vec);
    ~VecGame();

    void set_buffers(const std::vector<std::vector<void *>> &ac, const std::vector<std::vector<void *>> &ob, const std::vector<std::vector<void *>> &info, float *rew, uint8_t *first);
    void observe();
    void act();
    void wait_for_stepping_threads();
    
    // Thread-safe method to acquire next seed from shared list
    // Returns seed value, or -1 if all seeds exhausted
    int acquire_next_seed();

  private:
    // this mutex synchronizes access to pending_games and game->is_waiting_for_step
    // when game->is_waiting_for_step is set to true
    // ownership of game objects is transferred to the stepping thread until
    // game->is_waiting_for_step is set to false
    std::mutex stepping_thread_mutex;
    std::list<std::shared_ptr<Game>> pending_games;
    std::condition_variable pending_games_added;
    std::condition_variable pending_game_complete;
    std::vector<std::thread> threads;
    bool time_to_die = false;
};

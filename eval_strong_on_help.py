from pathlib import Path
import json
import os
import time
from typing import List, Optional, Dict, Any, Tuple

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
# We don't actually need the Evaluator for this task
from YRC.core.configs.global_configs import get_global_variable

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
import importlib


def rollout(policy, env, num_episodes, expected_seeds=None, padding_seeds=None):
    """
    Rollout the policy on the environment and collect episode returns.
    Using greedy=True by default for strong agent evaluation.
    
    Note: num_episodes does NOT need to be divisible by num_envs.
    We simply collect episodes until we have enough.
    
    Args:
        policy: Policy to evaluate
        env: Environment to run
        num_episodes: Number of episodes to collect
        expected_seeds: Set of seeds we expect to see (for validation)
        padding_seeds: Set of padding seeds that should NOT complete (for validation)
    """
    returns = []
    num_completed = 0
    target_episodes = num_episodes
    cumulative_rewards = [0.0] * env.num_envs
    
    # Track which seeds we've seen complete
    completed_seeds = []
    seed_to_return = {}
    
    # IMPORTANT: Calling reset() consumes the first batch of seeds in sequential mode
    # The initial reset triggers episodes that we should count
    print(f"  Calling reset() - this will start the first {env.num_envs} episodes")
    obs = env.reset()
    print(f"  Environment reset complete. Obs shape: {obs.shape}")
    
    # The environments are now running their first episodes using seeds 0-7
    # We need to count these episodes when they complete

    # Reset episode counter for heuristic policies
    if hasattr(policy, "reset_episode"):
        for i in range(env.num_envs):
            policy.reset_episode()

    step_count = 0
    last_debug_step = -1000
    stuck_counter = 0
    last_num_completed = 0
    
    while num_completed < target_episodes:
        if step_count % 100 == 0:
            print(f"  Step {step_count}: Completed {num_completed}/{target_episodes} episodes", end='\r')
        
        # Debug: if no progress for many steps, print detailed info
        if num_completed == last_num_completed:
            stuck_counter += 1
            if stuck_counter > 5000 and step_count > last_debug_step + 1000:
                print(f"\n  DEBUG: No progress for {stuck_counter} steps at {num_completed}/{target_episodes}")
                print(f"  DEBUG: done array = {done}")
                print(f"  DEBUG: cumulative_rewards = {[f'{r:.1f}' for r in cumulative_rewards]}")
                if 'level_seed' in info:
                    print(f"  DEBUG: level_seeds = {info.get('level_seed', 'N/A')}")
                if 'seeds_exhausted' in info:
                    print(f"  DEBUG: seeds_exhausted = {info.get('seeds_exhausted', 'N/A')}")
                last_debug_step = step_count
        else:
            stuck_counter = 0
            last_num_completed = num_completed
        
        action = policy.act(obs, greedy=True)
        next_obs, reward, done, info = env.step(action)

        for i in range(env.num_envs):
            cumulative_rewards[i] += reward[i]

            if done[i]:
                # Extract the level seed from info
                if isinstance(info, list):
                    level_seed = info[i].get('level_seed', None)
                elif isinstance(info, dict) and 'level_seed' in info:
                    # Some environments might return a single dict with arrays
                    level_seed = info['level_seed'][i] if hasattr(info['level_seed'], '__getitem__') else info['level_seed']
                else:
                    level_seed = None
                
                if num_completed < target_episodes:
                    returns.append(cumulative_rewards[i])
                    num_completed += 1
                    
                    # Track completed seed
                    if level_seed is not None:
                        completed_seeds.append(level_seed)
                        seed_to_return[level_seed] = cumulative_rewards[i]
                        
                        # Validation: check if this is a padding seed
                        if padding_seeds and level_seed in padding_seeds:
                            print(f"\n  WARNING: Padding seed {level_seed} completed! This shouldn't happen.")
                
                cumulative_rewards[i] = 0.0
                if hasattr(policy, "reset_episode"):
                    policy.reset_episode()
        
        obs = next_obs
        step_count += 1
    
    print()  # New line after progress
    
    # Validation: Check that all expected seeds completed exactly once
    if expected_seeds:
        completed_set = set(completed_seeds)
        expected_set = set(expected_seeds)
        
        # Check for missing seeds
        missing_seeds = expected_set - completed_set
        if missing_seeds:
            print(f"\n  ERROR: {len(missing_seeds)} expected seeds did not complete: {sorted(list(missing_seeds))[:10]}...")
        
        # Check for duplicate completions
        from collections import Counter
        seed_counts = Counter(completed_seeds)
        duplicates = {seed: count for seed, count in seed_counts.items() if count > 1}
        if duplicates:
            print(f"\n  ERROR: Some seeds completed multiple times: {duplicates}")
        
        # Check for unexpected seeds (not in expected set)
        unexpected_seeds = completed_set - expected_set
        if unexpected_seeds:
            print(f"\n  WARNING: {len(unexpected_seeds)} unexpected seeds completed: {sorted(list(unexpected_seeds))[:10]}...")
        
        print(f"\n  Validation summary:")
        print(f"    - Expected seeds: {len(expected_seeds)}")
        print(f"    - Completed seeds: {len(completed_set)}")
        print(f"    - Total episodes: {len(completed_seeds)}")
        print(f"    - All expected seeds completed exactly once: {missing_seeds == set() and duplicates == {} and unexpected_seeds == set()}")
    
    return returns, seed_to_return


def main():
    args = flags.make()
    # Don't set eval_mode = True since we're not creating a new experiment
    # We're just re-evaluating existing results
    
    # Validation
    if not args.npz_file:
        raise ValueError("Must provide --npz_file argument")
        
    # Check for strong agent argument
    strong_agent_path = getattr(args.agents, "strong", None)
    if not strong_agent_path:
        # Fallback to model_file if provided
        strong_agent_path = getattr(args, "model_file", None)
    
    if not strong_agent_path:
        raise ValueError("Must provide strong agent path via -strong <path>")
    
    # Load config
    config = config_utils.load(args.config, flags=args)
    
    # Record time for profiling purposes
    start_time = time.time()
    
    # Initialize wandb if available and not disabled
    use_wandb = os.environ.get("DISABLE_WANDB", "0") != "1"
    wandb_run = None
    
    if use_wandb:
        try:
            # Get job name from args or NPZ filename
            job_name = getattr(args, "name", None) or Path(args.npz_file).stem
            
            # Prepare wandb init parameters
            wandb_kwargs = {
                "name": f"strong_reval_{job_name}",
                "project": config.wandb.project or "yrc-bench-strong-reval",
                "group": config.wandb.group,
                "mode": config.wandb.mode,
                "job_type": "eval_strong",
                "config": {
                    "npz_file": args.npz_file,
                    "strong_agent": strong_agent_path,
                    "config_file": args.config,
                    "exp_name": config.exp_name,
                },
            }
            
            if config.wandb.entity is not None:
                wandb_kwargs["entity"] = config.wandb.entity
            
            wandb_run = wandb.init(**wandb_kwargs)
            print(f"\nInitialized wandb run: {wandb_run.name}")
        except ImportError:
            print("wandb not available, continuing without logging")
            use_wandb = False
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            use_wandb = False
    
    # Load NPZ results
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
    print(f"\nLoading results from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract data
    afhps = data['afhps']
    original_performances = data['performances']
    meta = data['meta']
    
    print(f"\nData summary:")
    print(f"  - Number of points: {len(afhps)}")
    print(f"  - AFHP range: {min(afhps):.2f}% - {max(afhps):.2f}%")
    print(f"  - Performance range: {min(original_performances):.2f} - {max(original_performances):.2f}")
    
    # Log to wandb
    if use_wandb and wandb_run:
        wandb.log({
            "num_points": len(afhps),
            "afhp_min": min(afhps),
            "afhp_max": max(afhps),
            "original_perf_min": min(original_performances),
            "original_perf_max": max(original_performances),
        })
    
    # We don't need an evaluator for this simple re-evaluation task
    
    # Import the specific environment module to get load_policy function
    benchmark = config.general.benchmark
    module = importlib.import_module(f"YRC.envs.{benchmark}")
    create_env_fn = getattr(module, "create_env")
    load_policy_fn = getattr(module, "load_policy")
    
    # Create a minimal environment just for loading the policy
    # We don't need agents loaded, just the environment structure
    print(f"\nBenchmark: {benchmark}")
    print(f"Loading strong agent from {strong_agent_path}...")
    
    # Create a dummy environment for policy initialization
    dummy_env = create_env_fn("train", config.environment)
    
    # Load the strong policy directly
    strong_policy = load_policy_fn(strong_agent_path, dummy_env)
    
    # Close dummy environment
    dummy_env.close()
    
    # First, get all unique seeds from all evaluation points
    print("\nCollecting unique seeds from all evaluation points...")
    all_seeds = set()
    split = "test"  # Default split
    
    for i, pt_meta in enumerate(meta):
        print(f"  Processing point {i+1}/{len(meta)}...", end='\r')
        summary_dict = pt_meta.get("summary", {})
        if split in summary_dict:
            level_seeds = summary_dict[split].get("level_seeds", [])
            all_seeds.update(level_seeds)
        else:
            # Try to find any split
            keys = list(summary_dict.keys())
            if keys:
                split = keys[0]
                level_seeds = summary_dict[split].get("level_seeds", [])
                all_seeds.update(level_seeds)
    
    print()  # New line after progress
    all_seeds = sorted(list(all_seeds))
    print(f"Found {len(all_seeds)} unique seeds across all evaluation points")
    print(f"Split: {split}")
    
    # Handle edge case where no seeds were found
    if len(all_seeds) == 0:
        print("\nWarning: No seeds found in any evaluation point!")
        print("This might indicate an issue with the NPZ file format.")
        # Let's examine the first meta entry to debug
        if len(meta) > 0:
            print("\nFirst meta entry structure:")
            print(f"  Keys: {list(meta[0].keys())}")
            if 'summary' in meta[0]:
                print(f"  Summary keys: {list(meta[0]['summary'].keys())}")
                if split in meta[0]['summary']:
                    print(f"  {split} keys: {list(meta[0]['summary'][split].keys())}")
    
    # Handle case where no seeds were found
    if len(all_seeds) == 0:
        print("\nNo seeds to evaluate. Creating dummy results...")
        strong_performances = [np.nan] * len(meta)
        total_time = time.time() - start_time
        
        # Save results even if empty
        output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
        np.savez(
            output_path,
            afhps=afhps,
            original_performances=original_performances,
            strong_performances=np.array(strong_performances),
            meta=meta
        )
        print(f"\nResults saved to {output_path}")
        return
    
    # Evaluate strong policy on ALL seeds
    print(f"\nEvaluating strong policy on all {len(all_seeds)} seeds...")
    print("="*60)
    
    # Create environment with all seeds
    # IMPORTANT: In sequential mode, we need to provide extra seeds to account for
    # the initial reset consuming some seeds. Add num_envs extra seeds.
    original_num_envs = config.environment.procgen.common.num_envs
    
    # Create padding seeds that are distinct from the original seeds
    max_seed = max(all_seeds) if all_seeds else 0
    padding_seeds = list(range(max_seed + 1000000, max_seed + 1000000 + original_num_envs))
    padded_seeds = all_seeds + padding_seeds
    
    print(f"Creating environment with {len(padded_seeds)} seeds ({len(all_seeds)} original + {len(padding_seeds)} padding)")
    print(f"  Original seeds range: {min(all_seeds)} - {max(all_seeds)}")
    print(f"  Padding seeds: {padding_seeds}")
    
    all_seeds_env = create_env_fn(
        split,
        config.environment,
        level_seeds=padded_seeds,
        level_seeds_mode="sequential"
    )
    
    print(f"Running {len(all_seeds)} episodes...")
    
    try:
        # Evaluate strong policy on all seeds with progress updates
        print(f"Starting rollout for {len(all_seeds)} episodes...")
        print(f"Environment has {all_seeds_env.num_envs} parallel environments")
        all_returns, seed_to_return_map = rollout(
            strong_policy, 
            all_seeds_env, 
            len(all_seeds),
            expected_seeds=all_seeds,
            padding_seeds=set(padding_seeds)
        )
        print(f"Rollout complete! Got {len(all_returns)} returns")
    finally:
        all_seeds_env.close()
    
    eval_time = time.time() - start_time
    print(f"\nCompleted evaluation in {eval_time:.1f}s")
    print(f"Mean return across all seeds: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    
    # Use the validated seed-to-return mapping from rollout
    seed_to_return = seed_to_return_map
    
    # Double-check we have returns for all seeds
    missing_returns = set(all_seeds) - set(seed_to_return.keys())
    if missing_returns:
        print(f"  ERROR: Missing returns for {len(missing_returns)} seeds: {sorted(list(missing_returns))[:10]}...")
    
    # Log overall statistics to wandb
    if use_wandb and wandb_run:
        wandb.log({
            "total_unique_seeds": len(all_seeds),
            "strong_agent_mean_return": np.mean(all_returns),
            "strong_agent_std_return": np.std(all_returns),
            "evaluation_time": eval_time,
        })
    
    # Now process each evaluation point and calculate mean return for help-requested seeds
    strong_performances = []
    
    print(f"\nCalculating strong performance for each evaluation point...")
    print("="*60)
    
    for i, pt_meta in enumerate(meta):
        # Access summary for the split
        summary_dict = pt_meta.get("summary", {})
        if split not in summary_dict:
            print(f"\n[{i+1}/{len(meta)}] Point {i}: No summary found for split '{split}', skipping...")
            strong_performances.append(np.nan)
            continue
                
        summary = summary_dict[split]
        level_seeds = summary.get("level_seeds", [])
        level_ood_pred = summary.get("level_ood_pred", [])
        
        # Identify help-requested seeds and get their returns
        help_returns = []
        for seed, pred in zip(level_seeds, level_ood_pred):
            if pred and seed in seed_to_return:
                help_returns.append(seed_to_return[seed])
        
        if not help_returns:
            print(f"\n[{i+1}/{len(meta)}] Point {i} (AFHP={afhps[i]:.2f}%): No help requested. Strong performance undefined.")
            strong_performances.append(np.nan)
            
            if use_wandb and wandb_run:
                wandb.log({
                    "point_idx": i,
                    "point_afhp": afhps[i],
                    "help_requested": False,
                    "num_help_seeds": 0,
                })
            continue
            
        # Calculate mean return for help-requested seeds
        mean_return = np.mean(help_returns)
        std_return = np.std(help_returns)
        strong_performances.append(mean_return)
        
        print(f"\n[{i+1}/{len(meta)}] Point {i} (AFHP={afhps[i]:.2f}%):")
        print(f"  - Original performance: {original_performances[i]:.2f}")
        print(f"  - Help requested on {len(help_returns)}/{len(level_seeds)} seeds ({len(help_returns)/len(level_seeds)*100:.1f}%)")
        print(f"  - Strong agent return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"  - Improvement over original: {mean_return - original_performances[i]:.2f}")
        
        # Log to wandb
        if use_wandb and wandb_run:
            wandb.log({
                "point_idx": i,
                "point_afhp": afhps[i],
                "help_requested": True,
                "num_help_seeds": len(help_returns),
                "total_seeds": len(level_seeds),
                "help_percentage": len(help_returns) / len(level_seeds) * 100,
                "original_performance": original_performances[i],
                "strong_performance": mean_return,
                "strong_performance_std": std_return,
                "improvement": mean_return - original_performances[i],
            })
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Re-evaluation completed!")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Unique seeds evaluated: {len(all_seeds)}")
    print(f"  - Points processed: {len(meta)}")
    print(f"  - Points with help requested: {sum(1 for p in strong_performances if not np.isnan(p))}")
    
    # Calculate summary statistics
    valid_strong_perfs = [p for p in strong_performances if not np.isnan(p)]
    if valid_strong_perfs:
        avg_strong = np.mean(valid_strong_perfs)
        avg_original = np.mean([original_performances[i] for i, p in enumerate(strong_performances) if not np.isnan(p)])
        avg_improvement = avg_strong - avg_original
        
        print(f"\nPerformance summary (for points with help):")
        print(f"  - Average original performance: {avg_original:.2f}")
        print(f"  - Average strong performance: {avg_strong:.2f}")
        print(f"  - Average improvement: {avg_improvement:.2f}")
    
    # Save results
    output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
    np.savez(
        output_path,
        afhps=afhps,
        original_performances=original_performances,
        strong_performances=np.array(strong_performances),
        meta=meta
    )
    print(f"\nResults saved to {output_path}")
    
    # Final wandb logging
    if use_wandb and wandb_run:
        wandb.log({
            "total_time": total_time,
            "total_points": len(meta),
            "unique_seeds_evaluated": len(all_seeds),
            "points_with_help": sum(1 for p in strong_performances if not np.isnan(p)),
        })
        
        if valid_strong_perfs:
            wandb.log({
                "avg_original_performance": avg_original,
                "avg_strong_performance": avg_strong,
                "avg_improvement": avg_improvement,
            })
        
        # Save output file as artifact
        artifact = wandb.Artifact(
            f"strong_reval_results_{job_name}",
            type="evaluation_results",
            description=f"Strong agent re-evaluation results for {job_name}",
        )
        artifact.add_file(str(output_path))
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    print(f"Time taken: {total_time:.1f} seconds")
    print(f"Total unique seeds evaluated: {len(all_seeds)}")


if __name__ == "__main__":
    main()
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


def rollout(policy, env, num_episodes):
    """
    Rollout the policy on the environment and collect episode returns.
    Using greedy=True by default for strong agent evaluation.
    
    Note: num_episodes does NOT need to be divisible by num_envs.
    We simply collect episodes until we have enough.
    """
    returns = []
    num_completed = 0
    target_episodes = num_episodes
    cumulative_rewards = [0.0] * env.num_envs
    
    obs = env.reset()

    # Reset episode counter for heuristic policies
    if hasattr(policy, "reset_episode"):
        for i in range(env.num_envs):
            policy.reset_episode()

    while num_completed < target_episodes:
        action = policy.act(obs, greedy=True)
        next_obs, reward, done, info = env.step(action)

        for i in range(env.num_envs):
            cumulative_rewards[i] += reward[i]

            if done[i]:
                if num_completed < target_episodes:
                    returns.append(cumulative_rewards[i])
                    num_completed += 1
                
                cumulative_rewards[i] = 0.0
                if hasattr(policy, "reset_episode"):
                    policy.reset_episode()
        
        obs = next_obs
        
    return returns


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
    
    # Process each point and re-evaluate strong agent on help-requested seeds
    strong_performances = []
    total_seeds_evaluated = 0
    
    print(f"\nStarting re-evaluation on {len(meta)} points...")
    print("="*60)
    
    for i, pt_meta in enumerate(meta):
        point_start_time = time.time()
        
        # Access summary for the split (usually 'test' since we ran AFHP on test)
        summary_dict = pt_meta.get("summary", {})
        split = "test"
        if split not in summary_dict:
            keys = list(summary_dict.keys())
            if keys:
                split = keys[0]
            else:
                print(f"\n[{i+1}/{len(meta)}] Point {i}: No summary found, skipping...")
                strong_performances.append(np.nan)
                continue
                
        summary = summary_dict[split]
        level_seeds = summary.get("level_seeds", [])
        level_ood_pred = summary.get("level_ood_pred", [])
        
        # Identify help-requested seeds
        help_seeds = [
            seed for seed, pred in zip(level_seeds, level_ood_pred) 
            if pred
        ]
        
        if not help_seeds:
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
            
        print(f"\n[{i+1}/{len(meta)}] Point {i} (AFHP={afhps[i]:.2f}%):")
        print(f"  - Original performance: {original_performances[i]:.2f}")
        print(f"  - Help requested on {len(help_seeds)}/{len(level_seeds)} seeds ({len(help_seeds)/len(level_seeds)*100:.1f}%)")
        print(f"  - Evaluating strong agent...")
        
        # Progress bar for rollout
        print(f"  - Running {len(help_seeds)} episodes...", end='', flush=True)
        rollout_start = time.time()
        
        # Create environment with only the help seeds
        help_env = create_env_fn(
            split,  # Use the same split as the original evaluation
            config.environment,
            level_seeds=help_seeds,
            level_seeds_mode="sequential"
        )
        
        try:
            # Use the original rollout function to evaluate
            returns = rollout(strong_policy, help_env, len(help_seeds))
            
            rollout_time = time.time() - rollout_start
            
            # Calculate statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
        finally:
            # Always close the environment
            help_env.close()
        strong_performances.append(mean_return)
        total_seeds_evaluated += len(help_seeds)
        
        print(f" Done! ({rollout_time:.1f}s)")
        print(f"  - Strong agent return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"  - Improvement over original: {mean_return - original_performances[i]:.2f}")
        
        point_time = time.time() - point_start_time
        
        # Log to wandb
        if use_wandb and wandb_run:
            wandb.log({
                "point_idx": i,
                "point_afhp": afhps[i],
                "help_requested": True,
                "num_help_seeds": len(help_seeds),
                "total_seeds": len(level_seeds),
                "help_percentage": len(help_seeds) / len(level_seeds) * 100,
                "original_performance": original_performances[i],
                "strong_performance": mean_return,
                "strong_performance_std": std_return,
                "improvement": mean_return - original_performances[i],
                "rollout_time": rollout_time,
                "point_processing_time": point_time,
                "cumulative_seeds_evaluated": total_seeds_evaluated,
            })
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Re-evaluation completed!")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Points processed: {len(meta)}")
    print(f"  - Total seeds evaluated: {total_seeds_evaluated}")
    print(f"  - Average time per point: {total_time/len(meta):.2f}s")
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
            "total_seeds_evaluated": total_seeds_evaluated,
            "points_with_help": sum(1 for p in strong_performances if not np.isnan(p)),
            "avg_time_per_point": total_time / len(meta),
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
    print(f"Total seeds evaluated: {total_seeds_evaluated}")


if __name__ == "__main__":
    main()
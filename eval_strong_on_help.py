from pathlib import Path
import json
import os
import time
from typing import List, Optional, Dict, Any, Tuple

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb


def main():
    args = flags.make()
    args.eval_mode = True
    
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
    
    # Create evaluator
    evaluator = Evaluator(config, config.environment)
    
    # Create strong agent policy
    # We need a dummy environment to load the policy
    dummy_envs = env_factory.make(config, None, None)
    strong_policy = policy_factory.make_from_checkpoint(
        strong_agent_path, 
        dummy_envs["train"], 
        config
    )
    
    # Close dummy environments
    for split_name in dummy_envs:
        dummy_envs[split_name].close()
    
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
        
        # Create environments with only the help seeds
        def make_help_envs():
            return env_factory.make(config, help_seeds, "sequential")
        
        # Evaluate strong agent on help seeds
        eval_result = evaluator.evaluate(
            policy=strong_policy,
            envs=make_help_envs(),
            split=split,
            num_episodes=len(help_seeds),
            base_log_path=None,  # Don't save videos for this
        )
        
        rollout_time = time.time() - rollout_start
        
        # Extract performance metrics
        mean_return = eval_result["performance"]["mean"]
        std_return = eval_result["performance"]["std"]
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
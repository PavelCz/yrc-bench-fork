
import flags
import YRC.core.configs.utils as config_utils
from YRC.core.configs.global_configs import get_global_variable
from pathlib import Path
import importlib
import numpy as np
import json
import logging
import time
from typing import List, Optional

# Re-use the rollout function from eval_policy if possible, or copy it. 
# copy is safer to avoid importing issues if eval_policy is strict.
# Actually, eval_policy is a script, not a module, so I cannot import from it easily.
# I will copy the rollout logic.

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
    
    # Validation
    if not args.npz_file:
        raise ValueError("Must provide --npz_file argument")
        
    # Check for strong agent argument
    # flags.py defines -strong mapping to --agents.strong
    # We check args.agents.strong
    strong_agent_path = getattr(args.agents, "strong", None)
    if not strong_agent_path:
        # Fallback to model_file if provided (though user said use -strong)
        strong_agent_path = getattr(args, "model_file", None)
    
    if not strong_agent_path:
        raise ValueError("Must provide strong agent path via -strong <path>")

    # Load config
    config = config_utils.load(args.config, flags=args)
    
    # Load NPZ results
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
    print(f"Loading results from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract data
    afhps = data['afhps']
    original_performances = data['performances']
    meta = data['meta']
    
    # Setup environment factory
    benchmark = config.general.benchmark
    print(f"Benchmark: {benchmark}")
    module = importlib.import_module(f"YRC.envs.{benchmark}")
    create_env_fn = getattr(module, "create_env")
    load_policy_fn = getattr(module, "load_policy")
    
    # Load Strong Policy
    # Create a dummy train env for loading policy
    print(f"Loading strong agent from {strong_agent_path}...")
    train_env = create_env_fn("train", config.environment)
    policy = load_policy_fn(strong_agent_path, train_env)
    train_env.close()
    
    strong_performances = []
    
    print(f"Starting re-evaluation on {len(meta)} points...")
    start_time = time.time()
    
    for i, pt_meta in enumerate(meta):
        # Depending on how meta is structured (dictionary or object)
        # In eval_afhp.py, we save pt.meta which is a dict
        
        # Access summary for the split (usually 'test' since we ran AFHP on test)
        # We need to find which split was used. AFHP usually runs on 'test'.
        # Let's check keys.
        summary_dict = pt_meta.get("summary", {})
        # Assume 'test' split. If not found, try keys.
        split = "test"
        if split not in summary_dict:
            keys = list(summary_dict.keys())
            if keys:
                split = keys[0]
            else:
                print(f"Warning: No summary found for point {i}, skipping...")
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
            print(f"Point {i}: No help requested. Strong performance undefined.")
            strong_performances.append(np.nan) # Or 0? Undefined is better.
            continue
            
        # Create environment with specific seeds
        # We need to ensure we run exactly len(help_seeds) episodes
        # And since we provide explicit seeds, we use sequential mode
        
        # Override config common.num_envs to match help_seeds length or a reasonable batch size
        # If help_seeds is large, we should batch. If small, we can run all at once or batch.
        # But create_env usually uses config.environment.common.num_envs
        # We should stick to the configured num_envs but run repeatedly until done.
        
        print(f"Point {i}: Evaluating strong agent on {len(help_seeds)} seeds...")
        
        # Create test env with ONLY the help seeds
        # We pass level_seeds explicitly
        test_env = create_env_fn(
            "test", 
            config.environment, 
            level_seeds=help_seeds, 
            level_seeds_mode="sequential"
        )
        
        try:
            returns = rollout(policy, test_env, len(help_seeds))
            mean_return = np.mean(returns)
            strong_performances.append(mean_return)
            print(f"  -> Mean Return: {mean_return:.2f}")
        finally:
            test_env.close()
            
    total_time = time.time() - start_time
    print(f"Re-evaluation finished in {total_time:.2f}s")
    
    # Save results
    output_path = npz_path.with_name(f"{npz_path.stem}_strong_reval.npz")
    np.savez(
        output_path,
        afhps=afhps,
        original_performances=original_performances,
        strong_performances=np.array(strong_performances),
        meta=meta
    )
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

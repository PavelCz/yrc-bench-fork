from pathlib import Path
import os
import time

import flags
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.core.configs.global_configs import get_global_variable

from YRC.policies.lightning_ae import LightningAEPolicy
from YRC.policies.mahalanobis_ae import MahalanobisAEPolicy
from YRC.policies.ood import OODPolicy
from YRC.policies.base import RandomPolicy

import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import List


def main():
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    # Record time for profiling purposes
    start_time = time.time()

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always" and not config.coord_policy.baseline:
        # If we are doing threshold search, the random alg does not need to train
        # anything. Thus, we do not need to load here.
        if config.general.algorithm != "random":
            policy.load_model(os.path.join(config.experiment_dir, config.file_name))

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config.evaluation)

    num_threshold_bins = args.eval.threshold_bins

    if num_threshold_bins < 5:
        raise ValueError("Number of threshold bins must be at least 5")

    # Initialize wandb logger
    save_dir = Path(str(get_global_variable("experiment_dir")))

    # Prepare wandb init parameters
    wandb_kwargs = {
        "name": config.exp_name,
        "project": config.wandb.project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "job_type": "train",
        "config": config,
    }

    if config.wandb.entity is not None:
        wandb_kwargs["entity"] = config.wandb.entity

    exp = wandb.init(**wandb_kwargs)

    wandb_logger = WandbLogger(
        save_dir=save_dir,
        experiment=exp,
    )

    split = "test"

    # Run the improved two-phase evaluation
    total_evals, all_results = run_two_phase_evaluation(
        policy, envs, split, wandb_logger, evaluator, num_threshold_bins
    )

    # Save result summary to file.
    log_file_path = get_global_variable("log_file")
    if log_file_path is None:
        raise ValueError(
            "Log file path is not set. Could not find path to save results."
        )
    log_file_path = Path(log_file_path)
    results_file_path = log_file_path.with_name(
        log_file_path.name.replace(".log", f"_{split}.npz")
    )
    
    # Extract data for saving
    thresholds = [result["threshold"] for result in all_results]
    summaries = [result["summary"] for result in all_results]
    afhp_values = [result["afhp"] for result in all_results]
    returns = [result["return"] for result in all_results]
    
    np.savez(
        results_file_path,
        thresholds=thresholds,
        summaries=np.array(summaries),
        afhp_values=afhp_values,
        returns=returns,
        results=np.array(summaries),  # Keep for backward compatibility
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {total_evals}")


def run_two_phase_evaluation(policy, envs, split, wandb_logger, evaluator, num_threshold_bins):
    """
    Improved two-phase evaluation approach:
    1. Systematic threshold sampling based on training percentiles
    2. Adaptive gap filling based on observed (AFHP, return) results
    """
    all_results = []
    
    # Phase 1: Systematic threshold sampling
    print("Phase 1: Systematic threshold sampling...")
    
    # Always start with extreme thresholds
    extreme_results = evaluate_extreme_thresholds(policy, envs, split, wandb_logger, evaluator)
    all_results.extend(extreme_results)
    
    # Get min/max returns for gap analysis
    min_return = min(result["return"] for result in extreme_results)
    max_return = max(result["return"] for result in extreme_results)
    print(f"Return range: {min_return:.3f} to {max_return:.3f}")
    
    # Sample thresholds systematically across training percentiles
    systematic_results = systematic_threshold_sampling(
        policy, envs, split, wandb_logger, evaluator, 
        num_threshold_bins - 2  # -2 because we already have extremes
    )
    all_results.extend(systematic_results)
    
    # Phase 2: Adaptive gap filling based on observed results
    print("Phase 2: Adaptive gap filling...")
    gap_results = adaptive_gap_filling(
        policy, envs, split, wandb_logger, evaluator,
        all_results, max_additional_evals=num_threshold_bins // 2
    )
    all_results.extend(gap_results)
    
    # Sort results by AFHP for easier analysis
    all_results.sort(key=lambda x: x["afhp"])
    
    total_evals = len(all_results)
    print(f"Phase 1 evals: {len(extreme_results) + len(systematic_results)}")
    print(f"Phase 2 evals: {len(gap_results)}")
    print(f"Total evals: {total_evals}")
    
    return total_evals, all_results


def evaluate_extreme_thresholds(policy, envs, split, wandb_logger, evaluator):
    """Evaluate extreme thresholds to establish bounds"""
    results = []
    
    # Threshold = inf (never ask for help, 0% AFHP)
    update_policy_params(policy, float("inf"))
    summary_inf = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=float("inf")
    )
    results.append({
        "threshold": float("inf"),
        "summary": summary_inf,
        "afhp": summary_inf[split]["action_1_frac"],
        "return": summary_inf[split]["reward_mean"],
        "method": "extreme_inf"
    })
    
    # Threshold = -inf (always ask for help, 100% AFHP)
    update_policy_params(policy, float("-inf"))
    summary_neg_inf = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=float("-inf")
    )
    results.append({
        "threshold": float("-inf"),
        "summary": summary_neg_inf,
        "afhp": summary_neg_inf[split]["action_1_frac"],
        "return": summary_neg_inf[split]["reward_mean"],
        "method": "extreme_neg_inf"
    })
    
    return results


def systematic_threshold_sampling(policy, envs, split, wandb_logger, evaluator, num_samples):
    """Sample thresholds systematically across training percentiles"""
    if num_samples <= 0:
        return []
    
    results = []
    
    # Sample percentiles uniformly, avoiding extremes
    percentiles = np.linspace(5, 95, num_samples)  # Avoid very extreme percentiles
    
    for i, percentile in enumerate(percentiles):
        try:
            # Get threshold for this training percentile
            threshold = policy.train_percentile(percentile)
            
            update_policy_params(policy, threshold)
            summary = evaluator.eval(
                policy, envs, [split], logger=wandb_logger, threshold=threshold
            )
            
            result = {
                "threshold": threshold,
                "summary": summary,
                "afhp": summary[split]["action_1_frac"],
                "return": summary[split]["reward_mean"],
                "method": "systematic",
                "train_percentile": percentile
            }
            results.append(result)
            print(f"Systematic {i+1}/{num_samples}: Train percentile={percentile:.1f}, "
                  f"Actual AFHP={result['afhp']:.3f}, Return={result['return']:.3f}")
            
        except Exception as e:
            print(f"Warning: Failed to get threshold for percentile {percentile}: {e}")
            continue
    
    return results


def adaptive_gap_filling(policy, envs, split, wandb_logger, evaluator, 
                        existing_results, max_additional_evals):
    """Fill gaps adaptively based on observed (AFHP, return) results"""
    if max_additional_evals <= 0:
        return []
    
    results = []
    current_results = existing_results.copy()
    
    for iteration in range(max_additional_evals):
        # Find the best threshold to try next based on current gaps
        best_threshold = find_best_threshold_for_gap_filling(policy, current_results)
        
        if best_threshold is None:
            print(f"No more thresholds to try, stopping after {iteration} iterations")
            break
            
        try:
            update_policy_params(policy, best_threshold)
            summary = evaluator.eval(
                policy, envs, [split], logger=wandb_logger, threshold=best_threshold
            )
            
            result = {
                "threshold": best_threshold,
                "summary": summary,
                "afhp": summary[split]["action_1_frac"],
                "return": summary[split]["reward_mean"],
                "method": "adaptive_gap_fill"
            }
            results.append(result)
            current_results.append(result)  # Update for next iteration
            
            print(f"Gap fill {iteration+1}: AFHP={result['afhp']:.3f}, Return={result['return']:.3f}")
            
        except Exception as e:
            print(f"Warning: Failed to evaluate threshold {best_threshold}: {e}")
            continue
    
    return results


def find_best_threshold_for_gap_filling(policy, existing_results):
    """
    Find the best threshold to try next for gap filling.
    Strategy: Look for the largest gaps in the 2D (AFHP, return) space 
    and try training percentiles that might fill those gaps.
    """
    if len(existing_results) < 2:
        return None
    
    # Sort results by AFHP
    sorted_results = sorted(existing_results, key=lambda x: x["afhp"])
    
    # Find the largest gap in the combined (AFHP, return) space
    best_gap_info = None
    best_gap_score = 0
    
    for i in range(len(sorted_results) - 1):
        curr_result = sorted_results[i]
        next_result = sorted_results[i + 1]
        
        # Calculate normalized gap size in 2D space
        afhp_gap = next_result["afhp"] - curr_result["afhp"]
        return_gap = abs(next_result["return"] - curr_result["return"])
        
        # Normalize gaps (simple approach - could be made more sophisticated)
        afhp_range = max(r["afhp"] for r in existing_results) - min(r["afhp"] for r in existing_results)
        return_range = max(r["return"] for r in existing_results) - min(r["return"] for r in existing_results)
        
        norm_afhp_gap = afhp_gap / afhp_range if afhp_range > 0 else 0
        norm_return_gap = return_gap / return_range if return_range > 0 else 0
        
        # Combined gap score (weighted sum)
        gap_score = norm_afhp_gap + 0.5 * norm_return_gap  # Weight AFHP gaps more heavily
        
        if gap_score > best_gap_score:
            best_gap_score = gap_score
            best_gap_info = {
                "left_result": curr_result,
                "right_result": next_result,
                "target_afhp": (curr_result["afhp"] + next_result["afhp"]) / 2,
                "afhp_gap": afhp_gap,
                "return_gap": return_gap
            }
    
    if best_gap_info is None:
        return None
    
    # Try to find a threshold that might produce AFHP in the gap
    # Strategy: If we know training percentiles for some existing results,
    # interpolate between them
    return interpolate_threshold_for_gap(policy, best_gap_info, existing_results)


def interpolate_threshold_for_gap(policy, gap_info, existing_results):
    """
    Try to find a threshold that might fill the identified gap.
    Uses interpolation of training percentiles when available.
    """
    left_result = gap_info["left_result"]
    right_result = gap_info["right_result"]
    
    # If both neighboring results have train_percentile info, interpolate
    if ("train_percentile" in left_result and "train_percentile" in right_result and
        left_result["train_percentile"] is not None and right_result["train_percentile"] is not None):
        
        # Interpolate percentile based on target AFHP position
        left_afhp = left_result["afhp"]
        right_afhp = right_result["afhp"]
        target_afhp = gap_info["target_afhp"]
        
        if right_afhp != left_afhp:
            alpha = (target_afhp - left_afhp) / (right_afhp - left_afhp)
            alpha = max(0, min(1, alpha))  # Clamp to [0, 1]
            
            target_percentile = (1 - alpha) * left_result["train_percentile"] + alpha * right_result["train_percentile"]
            
            try:
                return policy.train_percentile(target_percentile)
            except:
                pass
    
    # Fallback: Try a percentile roughly in the middle of unexplored space
    used_percentiles = set()
    for result in existing_results:
        if "train_percentile" in result and result["train_percentile"] is not None:
            used_percentiles.add(result["train_percentile"])
    
    # Find a percentile that hasn't been used yet
    for percentile in np.linspace(10, 90, 20):
        if percentile not in used_percentiles:
            try:
                return policy.train_percentile(percentile)
            except:
                continue
    
    return None


# Removed find_threshold_for_afhp function as it was based on incorrect assumption
# that we can directly target AFHP values. Instead, we sample thresholds and observe results.


def update_policy_params(policy, threshold):
    if isinstance(policy, LightningAEPolicy) or isinstance(policy, OODPolicy):
        policy.update_params({"threshold": threshold})
    elif isinstance(policy, RandomPolicy):
        if threshold == float("inf"):
            # An infinite threshold means that the policy will never ask for help.
            # We need to set the probability to 0.
            threshold = 0.0
        elif threshold == float("-inf"):
            # A negative infinite threshold means that the policy will always ask for help.
            # We need to set the probability to 1.
            threshold = 1.0
        policy.update_params(threshold)

    else:
        raise ValueError(
            f"Policy type {type(policy)} currently not supported for threshold search"
        )


if __name__ == "__main__":
    main()

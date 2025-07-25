from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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
from scipy.spatial import Delaunay
from pytorch_lightning.loggers import WandbLogger
import wandb


@dataclass
class EvalPoint:
    """Represents a single evaluation point in the (AFHP, return) space."""
    threshold: float
    train_percentile: float
    afhp: float  # Actual measured AFHP (action_1_frac)
    return_mean: float
    return_std: float
    summary: dict


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
            model_path = Path(config.experiment_dir) / config.file_name
            policy.load_model(str(model_path))

        # For the Mahalanobis AE, we need some additional initialization.
        if isinstance(policy, MahalanobisAEPolicy):
            policy.initialize_mahalanobis_detector(config)

    evaluator = Evaluator(config.evaluation)

    num_samples = args.eval.threshold_bins

    if num_samples < 5:
        raise ValueError("Number of samples must be at least 5")

    # Setup wandb logger
    wandb_logger = setup_wandb(config)
    
    # Run adaptive 2D sampling
    split = "test"
    eval_points = adaptive_2d_sampling(
        policy, evaluator, envs, wandb_logger, num_samples, split
    )

    # Convert to legacy format for compatibility
    results = convert_to_legacy_format(eval_points, num_samples)
    
    total_evals = len(eval_points)

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
    # Save both legacy format and new format
    np.savez(
        results_file_path,
        binned_train_percentiles=results["binned_train_percentiles"],
        binned_thresholds=results["binned_thresholds"],
        results=results["summaries"],
        # New format data
        eval_points=[{
            "threshold": p.threshold,
            "train_percentile": p.train_percentile,
            "afhp": p.afhp,
            "return_mean": p.return_mean,
            "return_std": p.return_std,
        } for p in eval_points]
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total evals: {total_evals}")


def setup_wandb(config):
    """Initialize wandb logger."""
    save_dir = Path(str(get_global_variable("experiment_dir")))
    
    wandb_kwargs = {
        "name": config.exp_name,
        "project": config.wandb.project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "job_type": "eval_thresholds",
        "config": config,
    }
    
    if config.wandb.entity is not None:
        wandb_kwargs["entity"] = config.wandb.entity
    
    exp = wandb.init(**wandb_kwargs)
    
    return WandbLogger(save_dir=save_dir, experiment=exp)


def evaluate_at_threshold(
    policy, evaluator, envs, threshold: float, wandb_logger, split: str
) -> dict:
    """Evaluate policy at a specific threshold."""
    update_policy_params(policy, threshold)
    summary = evaluator.eval(
        policy, envs, [split], logger=wandb_logger, threshold=threshold
    )
    return summary


def evaluate_boundaries(
    policy, evaluator, envs, wandb_logger, split: str
) -> Dict[str, EvalPoint]:
    """Phase 1: Establish bounds of the (AFHP, return) space."""
    boundary_points = {}
    
    # Always ask for help (threshold = -inf)
    threshold_always = float('-inf')
    summary_always = evaluate_at_threshold(
        policy, evaluator, envs, threshold_always, wandb_logger, split
    )
    boundary_points['always_help'] = EvalPoint(
        threshold=threshold_always,
        train_percentile=100.0,
        afhp=summary_always[split]['action_1_frac'],
        return_mean=summary_always[split]['reward_mean'],
        return_std=summary_always[split]['reward_std'],
        summary=summary_always
    )
    
    # Never ask for help (threshold = inf)
    threshold_never = float('inf')
    summary_never = evaluate_at_threshold(
        policy, evaluator, envs, threshold_never, wandb_logger, split
    )
    boundary_points['never_help'] = EvalPoint(
        threshold=threshold_never,
        train_percentile=0.0,
        afhp=summary_never[split]['action_1_frac'],
        return_mean=summary_never[split]['reward_mean'],
        return_std=summary_never[split]['reward_std'],
        summary=summary_never
    )
    
    print(f"Boundary evaluation complete:")
    print(f"  Always help: AFHP={boundary_points['always_help'].afhp:.3f}, "
          f"Return={boundary_points['always_help'].return_mean:.3f}")
    print(f"  Never help: AFHP={boundary_points['never_help'].afhp:.3f}, "
          f"Return={boundary_points['never_help'].return_mean:.3f}")
    
    return boundary_points


def evaluate_critical_percentiles(
    policy, evaluator, envs, wandb_logger, split: str
) -> List[EvalPoint]:
    """Phase 2: Evaluate at key percentiles."""
    critical_percentiles = [5, 10, 25, 50, 75, 90, 95]
    critical_points = []
    
    for percentile in critical_percentiles:
        threshold = policy.train_percentile(100 - percentile)
        summary = evaluate_at_threshold(
            policy, evaluator, envs, threshold, wandb_logger, split
        )
        
        point = EvalPoint(
            threshold=threshold,
            train_percentile=percentile,
            afhp=summary[split]['action_1_frac'],
            return_mean=summary[split]['reward_mean'],
            return_std=summary[split]['reward_std'],
            summary=summary
        )
        critical_points.append(point)
        
        print(f"Percentile {percentile}%: AFHP={point.afhp:.3f}, "
              f"Return={point.return_mean:.3f}")
    
    return critical_points


def compute_coverage_gaps(
    eval_points: List[EvalPoint], 
    return_scale: float = 1.0
) -> List[Dict]:
    """Identify gaps in 2D coverage using Delaunay triangulation."""
    if len(eval_points) < 3:
        return []
    
    # Extract coordinates
    points = np.array([[p.afhp, p.return_mean] for p in eval_points])
    
    # Normalize to [0, 1] for both axes
    afhp_range = points[:, 0].max() - points[:, 0].min()
    return_range = points[:, 1].max() - points[:, 1].min()
    
    if afhp_range == 0 or return_range == 0:
        return []
    
    normalized_points = np.zeros_like(points)
    normalized_points[:, 0] = (points[:, 0] - points[:, 0].min()) / afhp_range
    normalized_points[:, 1] = (points[:, 1] - points[:, 1].min()) / return_range
    
    # Compute triangulation
    try:
        tri = Delaunay(normalized_points)
    except:
        # If triangulation fails, return empty list
        return []
    
    # Calculate triangle areas and centers
    gaps = []
    for simplex in tri.simplices:
        # Get triangle vertices
        triangle = normalized_points[simplex]
        
        # Calculate area using cross product
        v1 = triangle[1] - triangle[0]
        v2 = triangle[2] - triangle[0]
        area = 0.5 * abs(np.cross(v1, v2))
        
        # Calculate centroid
        centroid = triangle.mean(axis=0)
        
        # Denormalize centroid
        centroid_afhp = centroid[0] * afhp_range + points[:, 0].min()
        centroid_return = centroid[1] * return_range + points[:, 1].min()
        
        gaps.append({
            'area': area,
            'centroid': (centroid_afhp, centroid_return),
            'vertices': simplex
        })
    
    return sorted(gaps, key=lambda x: x['area'], reverse=True)


def estimate_threshold_for_target(
    target_afhp: float,
    eval_points: List[EvalPoint],
    policy
) -> float:
    """Estimate threshold needed to achieve target AFHP."""
    # Since AFHP is monotonic with respect to percentile,
    # we can use interpolation in percentile space
    
    # Sort points by AFHP
    sorted_points = sorted(eval_points, key=lambda p: p.afhp)
    
    # Find bracketing points
    left_point = None
    right_point = None
    
    for i in range(len(sorted_points) - 1):
        if sorted_points[i].afhp <= target_afhp <= sorted_points[i + 1].afhp:
            left_point = sorted_points[i]
            right_point = sorted_points[i + 1]
            break
    
    if left_point is None:
        # Target is outside the range
        if target_afhp < sorted_points[0].afhp:
            # Need higher threshold (lower percentile)
            return policy.train_percentile(100 - sorted_points[0].train_percentile / 2)
        else:
            # Need lower threshold (higher percentile)
            last_percentile = sorted_points[-1].train_percentile
            new_percentile = last_percentile + (100 - last_percentile) / 2
            return policy.train_percentile(100 - new_percentile)
    
    # Linear interpolation in percentile space
    alpha = (target_afhp - left_point.afhp) / (right_point.afhp - left_point.afhp)
    interp_percentile = (1 - alpha) * left_point.train_percentile + alpha * right_point.train_percentile
    
    return policy.train_percentile(100 - interp_percentile)


def refine_return_axis(
    eval_points: List[EvalPoint],
    policy,
    evaluator,
    envs,
    wandb_logger,
    split: str,
    remaining_budget: int
) -> List[EvalPoint]:
    """Focus on filling gaps in return axis for smooth curves."""
    if remaining_budget <= 0:
        return eval_points
    
    # Sort points by AFHP
    sorted_points = sorted(eval_points, key=lambda p: p.afhp)
    
    # Find largest return jumps between adjacent AFHP values
    return_gaps = []
    for i in range(len(sorted_points) - 1):
        p1, p2 = sorted_points[i], sorted_points[i+1]
        afhp_diff = abs(p2.afhp - p1.afhp)
        return_diff = abs(p2.return_mean - p1.return_mean)
        
        if afhp_diff > 0.01:  # Only consider if AFHP difference is significant
            # Normalize by AFHP difference to find steep changes
            steepness = return_diff / afhp_diff
            return_gaps.append({
                'steepness': steepness,
                'return_diff': return_diff,
                'p1': p1,
                'p2': p2,
                'mid_afhp': (p1.afhp + p2.afhp) / 2,
                'mid_return': (p1.return_mean + p2.return_mean) / 2
            })
    
    # Sort by steepness/return difference
    return_gaps.sort(key=lambda x: x['steepness'], reverse=True)
    
    # Fill the largest gaps
    new_points = []
    for i in range(min(remaining_budget, len(return_gaps))):
        gap = return_gaps[i]
        
        # Try percentile between the two points
        mid_percentile = (gap['p1'].train_percentile + gap['p2'].train_percentile) / 2
        threshold = policy.train_percentile(100 - mid_percentile)
        
        # Evaluate
        summary = evaluate_at_threshold(
            policy, evaluator, envs, threshold, wandb_logger, split
        )
        
        new_point = EvalPoint(
            threshold=threshold,
            train_percentile=mid_percentile,
            afhp=summary[split]['action_1_frac'],
            return_mean=summary[split]['reward_mean'],
            return_std=summary[split]['reward_std'],
            summary=summary
        )
        
        new_points.append(new_point)
        print(f"Refinement {i+1}: AFHP={new_point.afhp:.3f}, "
              f"Return={new_point.return_mean:.3f} (gap steepness={gap['steepness']:.2f})")
    
    return eval_points + new_points


def adaptive_2d_sampling(
    policy,
    evaluator,
    envs,
    wandb_logger,
    num_samples: int,
    split: str,
    min_gap_size: float = 0.01
) -> List[EvalPoint]:
    """Main adaptive sampling algorithm."""
    
    # Phase 1: Boundary evaluation
    print("\nPhase 1: Evaluating boundaries...")
    eval_points = []
    boundary_points = evaluate_boundaries(policy, evaluator, envs, wandb_logger, split)
    eval_points.extend(boundary_points.values())
    
    # Phase 2: Critical percentiles
    print("\nPhase 2: Evaluating critical percentiles...")
    critical_points = evaluate_critical_percentiles(
        policy, evaluator, envs, wandb_logger, split
    )
    eval_points.extend(critical_points)
    
    # Phase 3: Adaptive gap filling
    print("\nPhase 3: Adaptive gap filling...")
    samples_taken = len(eval_points)
    
    while samples_taken < num_samples:
        # Find coverage gaps
        gaps = compute_coverage_gaps(eval_points)
        
        # Filter out small gaps
        significant_gaps = [g for g in gaps if g['area'] > min_gap_size]
        
        if not significant_gaps:
            print("No significant gaps found. Moving to return axis refinement.")
            break
        
        # Take the largest gap
        largest_gap = significant_gaps[0]
        target_afhp, target_return = largest_gap['centroid']
        
        # Estimate threshold for this target
        threshold = estimate_threshold_for_target(
            target_afhp, eval_points, policy
        )
        
        # Evaluate at this threshold
        summary = evaluate_at_threshold(
            policy, evaluator, envs, threshold, wandb_logger, split
        )
        
        # Create new eval point
        # Get percentile from threshold (approximate)
        train_percentiles = [p.train_percentile for p in eval_points]
        thresholds = [p.threshold for p in eval_points if not np.isinf(p.threshold)]
        if thresholds and not np.isinf(threshold):
            # Estimate percentile by interpolation
            sorted_idx = np.argsort(thresholds)
            sorted_thresholds = np.array(thresholds)[sorted_idx]
            sorted_percentiles = np.array([train_percentiles[i] for i in sorted_idx])
            estimated_percentile = np.interp(threshold, sorted_thresholds[::-1], sorted_percentiles[::-1])
        else:
            estimated_percentile = 50.0  # Default
        
        new_point = EvalPoint(
            threshold=threshold,
            train_percentile=estimated_percentile,
            afhp=summary[split]['action_1_frac'],
            return_mean=summary[split]['reward_mean'],
            return_std=summary[split]['reward_std'],
            summary=summary
        )
        
        eval_points.append(new_point)
        samples_taken += 1
        
        print(f"Sample {samples_taken}/{num_samples}: "
              f"AFHP={new_point.afhp:.3f}, "
              f"Return={new_point.return_mean:.3f} "
              f"(gap area={largest_gap['area']:.4f})")
    
    # Phase 4: Return axis refinement
    remaining_budget = num_samples - samples_taken
    if remaining_budget > 0:
        print(f"\nPhase 4: Return axis refinement ({remaining_budget} samples)...")
        eval_points = refine_return_axis(
            eval_points, policy, evaluator, envs, wandb_logger, split, remaining_budget
        )
    
    return eval_points


def convert_to_legacy_format(
    eval_points: List[EvalPoint], 
    num_bins: int
) -> Dict:
    """Convert new format to legacy format for compatibility."""
    # Sort by AFHP for consistent ordering
    sorted_points = sorted(eval_points, key=lambda p: p.afhp)
    
    # Create bins
    afhp_bins = np.linspace(0, 100, num_bins + 1)
    binned_train_percentiles = [None] * num_bins
    binned_thresholds = [None] * num_bins
    summaries = [None] * num_bins
    
    # Assign points to bins
    for point in sorted_points:
        afhp_percent = point.afhp * 100
        for i in range(len(afhp_bins) - 1):
            if afhp_bins[i] <= afhp_percent <= afhp_bins[i + 1]:
                if summaries[i] is None:  # Only use first point in each bin
                    summaries[i] = point.summary
                    binned_thresholds[i] = point.threshold
                    binned_train_percentiles[i] = point.train_percentile
                break
    
    # Fill empty bins with interpolation
    for i in range(num_bins):
        if summaries[i] is None:
            # Find nearest non-empty bins
            left_idx = i - 1
            while left_idx >= 0 and summaries[left_idx] is None:
                left_idx -= 1
            
            right_idx = i + 1
            while right_idx < num_bins and summaries[right_idx] is None:
                right_idx += 1
            
            if left_idx >= 0 and right_idx < num_bins:
                # Interpolate
                alpha = (i - left_idx) / (right_idx - left_idx)
                summaries[i] = summaries[left_idx]  # Use left neighbor's summary
                binned_train_percentiles[i] = (
                    (1 - alpha) * binned_train_percentiles[left_idx] + 
                    alpha * binned_train_percentiles[right_idx]
                )
                binned_thresholds[i] = binned_thresholds[left_idx]  # Use left threshold
            elif left_idx >= 0:
                # Use left neighbor
                summaries[i] = summaries[left_idx]
                binned_train_percentiles[i] = binned_train_percentiles[left_idx]
                binned_thresholds[i] = binned_thresholds[left_idx]
            elif right_idx < num_bins:
                # Use right neighbor
                summaries[i] = summaries[right_idx]
                binned_train_percentiles[i] = binned_train_percentiles[right_idx]
                binned_thresholds[i] = binned_thresholds[right_idx]
    
    return {
        "summaries": np.array(summaries),
        "binned_train_percentiles": binned_train_percentiles,
        "binned_thresholds": binned_thresholds
    }


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

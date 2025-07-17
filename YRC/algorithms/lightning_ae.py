import logging
from typing import Dict, List, Optional, Any

from YRC.core import Algorithm
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.configs.utils import config_logging
from YRC.policies.lightning_ae import LightningAEPolicy
import torch


class AutoencoderAlgorithm(Algorithm):
    """
    Algorithm for training and evaluating autoencoder models using PyTorch Lightning.
    Uses LightningAEPolicy to wrap the actual Lightning module, similar to how
    OODAlgorithm uses OODPolicy.

    This algorithm can work with different types of autoencoders including:
    - Basic Autoencoder
    - VAE variants
    - Other autoencoder architectures available in the vae_models registry
    """

    def __init__(self, config: Any, env: Any) -> None:
        super().__init__()
        self.args = config
        self.env = env
        self.save_dir: str = get_global_variable("experiment_dir")

    def train(
        self,
        policy: LightningAEPolicy,
        envs: Optional[Dict[str, Any]],
        rollout_obs: List[torch.Tensor],
        evaluator: Optional[Any] = None,
        train_split: Optional[str] = None,
        eval_splits: Optional[List[str]] = None,
        do_threshold_search: bool = True,
    ):
        """
        Train an autoencoder model using PyTorch Lightning and LightningAEPolicy.
        Now follows the same pattern as OODAlgorithm.

        Args:
            policy: LightningAEPolicy instance
            envs: Environment dictionary
            evaluator: Evaluator for testing performance
            train_split: Training split name
            eval_splits: List of evaluation split names
            do_threshold_search: Whether to search for optimal threshold
        """
        args = self.args

        log_file = get_global_variable("log_file")

        config_logging(log_file)

        if do_threshold_search:
            raise NotImplementedError(
                "Threshold search not implemented for Lightning AE"
            )

        # Initialize Lightning AE detector (similar to OOD algorithm)
        policy.initialize_ood_detector(args, envs["train"])

        logging.info(
            f"Gathering {args.num_rollouts} rollouts for training OOD detector."
        )

        num_rollouts_test = max(args.num_rollouts // 10, 1)
        # Ensure that num_rollouts_test is divisible by envs["train"].num_envs.
        if num_rollouts_test % envs["train"].num_envs != 0:
            num_rollouts_test += (
                envs["train"].num_envs - num_rollouts_test % envs["train"].num_envs
            )

        logging.info(f"Collected training dataset of shape {len(rollout_obs)}")

        logging.info("Starting training OOD detector.")

        # Train OOD detector
        policy.fit(x=rollout_obs, x_threshold=rollout_obs)

        # TODO: Implement autoencoder validation set

        logging.info("Saving trained model.")
        policy.save_model("trained", self.save_dir)

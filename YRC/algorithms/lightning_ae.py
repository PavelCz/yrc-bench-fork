import logging
from typing import Dict, List, Optional, Any

from YRC.core import Algorithm
from YRC.core.configs.global_configs import get_global_variable
from YRC.policies.lightning_ae import LightningAEPolicy


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

        if do_threshold_search:
            raise NotImplementedError(
                "Threshold search not implemented for Lightning AE"
            )

        # Initialize Lightning AE detector (similar to OOD algorithm)
        policy.initialize_ood_detector(args, envs["train"])

        # Generate rollouts for training OOD detector
        rollout_obs = policy.gather_rollouts(envs["train"], args.num_rollouts)
        rollout_obs_threshold = policy.gather_rollouts(envs["train"], args.num_rollouts)

        # Train OOD detector
        policy.fit(x=rollout_obs, x_threshold=rollout_obs_threshold)

        # TODO: Implement autoencoder validation set

        logging.info("Saving trained model.")
        policy.save_model("trained", self.save_dir)

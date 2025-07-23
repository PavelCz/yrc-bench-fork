import os
import logging
import numpy as np
from typing import List, Optional, Any, Tuple, Union, cast
import sys
from YRC.core.configs import ConfigDict
import torch
from scipy.spatial.distance import mahalanobis

from YRC.core.configs.global_configs import get_global_variable
from YRC.core.utils import to_tensor, load_rollouts_from_file
from YRC.policies.lightning_ae import LightningAEPolicy

from pytorch_lightning.loggers import WandbLogger

# Add pytorch_vae to Python path.
# This is somewhat awkward, but necessary since we're inlcuding pytorch_vae as a git
# submodule.
pytorch_vae_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "lib", "pytorch_vae"
)
if pytorch_vae_path not in sys.path:
    sys.path.insert(0, pytorch_vae_path)


class MahalanobisAEPolicy(LightningAEPolicy):
    """
    Policy wrapper for PyTorch Lightning autoencoder models that provides OOD detection
    capabilities through reconstruction error scoring.

    Extends OODPolicy to reuse common functionality while providing Lightning-specific
    autoencoder implementation.
    """

    def __init__(
        self, config: Any, env: Any, logger: Optional[WandbLogger] = None
    ) -> None:
        # Initialize parent class
        super().__init__(config, env, logger=logger)

        # Alpha and beta are mixing parameters for the novelty loss.
        # Alpha is the weight of the Mahalanobis distance, beta is the weight of the
        # reconstruction loss.
        self.alpha: float = 0.5
        self.beta: float = 0.5

        self.mean_vector: Optional[np.ndarray] = None
        self.inv_cov_matrix: Optional[np.ndarray] = None

    def initialize_mahalanobis_detector(self, config: ConfigDict) -> None:
        # Determine mean vector of the encoded training set.
        training_set = load_rollouts_from_file(config)

        # Encode the training set.
        aggregated_vector = None
        count = 0
        for obs in training_set:
            obs = obs.to(self.device)
            # Encode returns a list of length 1, so we need to index into it.
            encoded_obs = self.clf.encode(obs.unsqueeze(0))[0]
            if aggregated_vector is None:
                aggregated_vector = encoded_obs
            else:
                aggregated_vector += encoded_obs
            count += 1

        # Divide by the number of observations to get the mean.
        aggregated_vector /= count

        # Store the mean vector.
        self.mean_vector = aggregated_vector[0].cpu().detach().numpy()

        # Encode the training set.
        encoded_training_set = []
        for obs in training_set:
            obs = obs.to(self.device)
            encoded_obs = self.clf.encode(obs.unsqueeze(0))[0].squeeze(0)
            encoded_training_set.append(encoded_obs.cpu())

        training_set_tensor = torch.stack(encoded_training_set).detach().numpy()
        # Flatten images into vectors.
        cov_matrix = np.cov(
            training_set_tensor, rowvar=False
        )  # rowvar=False treats rows as samples
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)

        # TODO: Determine alpha and beta parameters.

        # since the Mahalanobis is an augmentation of a base AE-based OOD detector,
        # we also need to update the training set decision scores.
        train_decision_scores = self._compute_decision_scores(
            torch.stack(training_set), return_recons=False
        )
        self._train_decision_scores = cast(np.ndarray, train_decision_scores)

    def _compute_decision_scores(
        self, x: torch.Tensor, return_recons: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """Compute OOD scores on the training data. This combines
        the reconstruction error from the autoencoder and the Mahalanobis distance."""
        ret = super()._compute_decision_scores(x, return_recons=return_recons)
        if return_recons:
            recon_scores, recons = ret
        else:
            recon_scores = ret

        x = x.to(self.device)
        # Compute the Mahalanobis distance.
        mahalanobis_distance = self._compute_mahalanobis_distance(x)

        scores = self.alpha * mahalanobis_distance + self.beta * recon_scores

        decision_scores = np.array(scores)

        if return_recons:
            return decision_scores, recons
        return decision_scores

    def act(
        self,
        obs: dict,
        greedy: bool = False,
        return_scores_and_recons: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
        # This is pretty much a copy of the act method in the parent LightningAEPolicy
        # class, but we use our overriden _compute_decision_scores method instead to
        # include the Mahalanobis distance in the decision scores.
        keys = {
            "obs": ["env_obs"],
            "hidden": ["weak_features"],
            "dist": ["weak_logit"],
            "hidden_obs": ["env_obs", "weak_features"],
            "hidden_dist": ["weak_features", "weak_logit"],
            "obs_dist": ["env_obs", "weak_logit"],
            "obs_hidden_dist": ["env_obs", "weak_features", "weak_logit"],
        }[self.feature_type]

        if get_global_variable("benchmark") in ["cliport", "minigrid"]:
            observation = [
                to_tensor(
                    obs[key]["image"] if key == "env_obs" else to_tensor(obs[key])
                )
                for key in keys
            ]
        else:
            observation = [to_tensor(obs[key]) for key in keys]

        if self.feature_type in ["obs", "hidden", "dist"]:
            observation = observation[0]

        # Get decision score for the observation.
        if return_scores_and_recons:
            recon_loss, recons = self._compute_decision_scores(
                observation, return_recons=return_scores_and_recons
            )
        else:
            recon_loss = self._compute_decision_scores(observation)

        # Use our own threshold instead of self.clf.threshold_
        action: np.ndarray = (recon_loss > self.threshold_).astype(int)

        if not np.any(action == 0) and not np.any(action == 1):
            logging.warning("No action selected as normal or OOD")

        if return_scores_and_recons:
            return action, recon_loss, recons

        return action

    def _compute_mahalanobis_distance(self, obs: torch.Tensor) -> np.ndarray:
        # Encode the observation.
        # Encode returns a list of length 1, so we need to index into it.
        encoded_obs = self.clf.encode(obs)[0].cpu().detach().numpy()

        mahalanobis_distances = []
        for i in range(len(encoded_obs)):
            # Compute the Mahalanobis distance.
            mahalanobis_distance = mahalanobis(
                encoded_obs[i], self.mean_vector, self.inv_cov_matrix
            )
            mahalanobis_distances.append(mahalanobis_distance)

        return np.array(mahalanobis_distances)

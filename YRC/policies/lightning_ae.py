import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys


from YRC.policies.ood import OODPolicy
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.dataset import ObservationDataset, ObservationDataModule
from YRC.core.utils import to_tensor

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import yaml

# Add pytorch_vae to Python path.
# This is somewhat awkward, but necessary since we're inlcuding pytorch_vae as a git
# submodule.
pytorch_vae_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "lib", "pytorch_vae"
)
if pytorch_vae_path not in sys.path:
    sys.path.insert(0, pytorch_vae_path)

# Import VAE components
from lib.pytorch_vae.models import vae_models  # noqa: E402
from lib.pytorch_vae.experiment import VAEXperiment  # noqa: E402
from lib.pytorch_vae.models.base import BaseVAE  # noqa: E402


class LightningAEPolicy(OODPolicy):
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
        super().__init__(config, env)

        # Lightning-specific attributes - use self.model instead of self.clf
        self.clf: Optional[BaseVAE] = None
        self.experiment: Optional[VAEXperiment] = None

        self.batch_size = None

        # OOD detection attributes
        self.threshold_: float = 0.0
        self._train_decision_scores: Optional[np.ndarray] = None

        self.runner: Optional[Trainer] = None
        self.data_config: Optional[Dict[str, Any]] = None

        self.logger = None

    def initialize_ood_detector(self, args: Any, env: Any) -> None:
        """Initialize the Lightning autoencoder model."""
        dummy_obs: Dict[str, Any] = env.reset()

        # Determine input shape based on feature type
        feature_type_to_shapes = {
            "obs": lambda dummy_obs: (
                dummy_obs["env_obs"]["image"]
                if get_global_variable("benchmark") in ["cliport", "minigrid"]
                else dummy_obs["env_obs"]
            ).shape,
            "hidden": lambda dummy_obs: dummy_obs["weak_features"].shape,
            "hidden_obs": lambda dummy_obs: (
                (
                    dummy_obs["env_obs"]["image"]
                    if get_global_variable("benchmark") in ["cliport", "minigrid"]
                    else dummy_obs["env_obs"]
                ).shape
                + dummy_obs["weak_features"].shape[1:]
            ),
            "dist": lambda dummy_obs: dummy_obs["weak_logit"].shape,
            "hidden_dist": lambda dummy_obs: (
                dummy_obs["weak_features"].shape + dummy_obs["weak_logit"].shape[1:]
            ),
            "obs_dist": lambda dummy_obs: (
                (
                    dummy_obs["env_obs"]["image"]
                    if get_global_variable("benchmark") in ["cliport", "minigrid"]
                    else dummy_obs["env_obs"]
                ).shape
                + dummy_obs["weak_logit"].shape[1:]
            ),
            "obs_hidden_dist": lambda dummy_obs: (
                (
                    dummy_obs["env_obs"]["image"]
                    if get_global_variable("benchmark") in ["cliport", "minigrid"]
                    else dummy_obs["env_obs"]
                ).shape
                + dummy_obs["weak_features"].shape[1:]
                + dummy_obs["weak_logit"].shape[1:]
            ),
        }

        dummy_obs_shape: Tuple[int, ...] = feature_type_to_shapes[self.feature_type](
            dummy_obs
        )

        # FYI the passed args are the algorithm args, self.args are the policy args.

        self.batch_size = args.batch_size

        epochs = args.epoch

        # Adjust model config based on input shape
        # if len(dummy_obs_shape) > 2:  # Image data
        #     self.model_config["in_channels"] = (
        #         dummy_obs_shape[1] if len(dummy_obs_shape) == 4 else dummy_obs_shape[0]
        #     )
        method_name = self.args.method
        model_config_path = args.model_config_path
        with open(model_config_path, "r") as file:
            try:
                model_config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        # For later dataset initialization.
        self.data_config = model_config["data_params"].copy()
        # Remove params we don't need. I'd rather be explicit about this than drop
        # things silently.
        self.data_config.pop("data_path")

        # Optionally override parameters.
        if args.batch_size is not None:
            self.data_config["train_batch_size"] = args.batch_size
            self.data_config["val_batch_size"] = args.batch_size

        # Initialize model
        if method_name not in vae_models:
            raise ValueError(
                f"Model {method_name} not found in vae_models registry. "
                f"Available models: {list(vae_models.keys())}"
            )

        model_params = model_config["model_params"]

        # Do optional overrides.
        if self.args.latent_dim is not None:
            model_params["latent_dim"] = self.args.latent_dim

        self.clf = vae_models[method_name](**model_params)

        save_dir = Path(str(get_global_variable("experiment_dir")))

        if self.logger is None:
            self.logger = TensorBoardLogger(save_dir=save_dir, name=self.args.exp_name)

        # First, define some default params. This is for exp_params.
        exp_params = {
            "dont_annotate_loss": False,
            "extra_image_outputs": False,
            "test_output_dir": str(save_dir / "full_test_output"),
            "histogram_only": False,
        }
        new_exp_params = model_config["exp_params"]
        exp_params.update(new_exp_params)

        self.experiment = VAEXperiment(self.clf, exp_params)

        self.experiment.to(self.device)

        save_dir = get_global_variable("experiment_dir")
        if save_dir is None:
            raise ValueError("Experiment directory is not set")

        if self.device.type == "cuda":
            accelerator = "auto"
            num_gpus = 1
        elif self.device.type == "cpu":
            accelerator = "cpu"
            num_gpus = 0
        else:
            raise ValueError(f"Invalid device type: {self.device.type}")

        self.runner = Trainer(
            logger=self.logger,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(
                    save_top_k=1,
                    dirpath=str(Path(save_dir) / "checkpoints"),
                    filename="best",
                    monitor="val_loss",
                    save_last=True,
                ),
            ],
            strategy=DDPPlugin(find_unused_parameters=False),
            max_epochs=epochs,
            accelerator=accelerator,
            # devices=self.device.index,
            gpus=num_gpus,
        )

        # Move to device
        self.clf.to(self.device)

        # Set clf_name for compatibility but use self.model instead of self.clf
        self.clf_name = "LightningAE"

        logging.info(f"Initialized Lightning AE model: {method_name}")
        logging.info(f"Input shape: {dummy_obs_shape}")
        logging.info(f"Latent dim: {model_config['model_params']['latent_dim']}")

    def fit(
        self, x: torch.Tensor, x_threshold: torch.Tensor, y: Optional[Any] = None
    ) -> None:
        """
        Override parent fit method to work with Lightning model instead of self.clf.
        """
        logging.info(
            "Lightning AE Policy: Computing decision scores for threshold setting"
        )

        # x = x.to(self.device)
        # x_threshold = x_threshold.to(self.device)

        # Turn sequence of observations x into a dataset
        train_dataset = ObservationDataset(x)
        test_dataset = ObservationDataset(x_threshold)

        # Remove args that we override.
        self.data_config.pop("num_workers")

        datamodule = ObservationDataModule(
            **self.data_config,
            # Reset num workers since this was causing issues.
            num_workers=0,
            train_dataset_torch=train_dataset,
            test_dataset_torch=test_dataset,
        )

        self.runner.fit(self.experiment, datamodule=datamodule)

        # Run test run to generate samples. Uses test dataset from datamodule as
        # specified above.
        self.runner.test(self.experiment, datamodule=datamodule)

        # Compute decision scores for threshold setting
        self._train_decision_scores = self._compute_decision_scores(x)

    def _compute_decision_scores(self, x: torch.Tensor) -> np.ndarray:
        """Compute reconstruction error scores on the training data."""
        if self.clf is None:
            raise ValueError("Model not initialized")

        self.clf.eval()

        self.clf.to(self.device)

        scores: List[float] = []

        with torch.no_grad():
            # Go through each observation in the dataset.
            for i in range(0, len(x)):
                img: torch.Tensor = x[i].to(self.device)
                # Add a batch dimension.
                batch = img.unsqueeze(0)

                # Handle different input types
                # if len(batch.shape) == 2:  # Flattened data
                #     # Reshape to appropriate image format if needed
                #     if self.feature_type == "obs":
                #         # Assume square images, infer dimensions
                #         size: int = int(
                #             np.sqrt(batch.shape[1] // self.model_config["in_channels"])
                #         )
                #         batch = batch.view(
                #             -1, self.model_config["in_channels"], size, size
                #         )

                # Get reconstruction
                if hasattr(self.clf, "forward"):
                    # In case, e.g., there are skip connections, we want to use the full
                    # forward pass.
                    reconstruction: torch.Tensor
                    reconstruction, _input, recons_features, input_features = self.clf(
                        batch
                    )
                else:
                    reconstruction = self.clf.decode(self.clf.encode(batch)[0])
                    recons_features = None
                    input_features = None

                # Use the models loss function to compute the reconstruction error.
                # recons_features and input_features are only used by VGG loss
                loss_dict = self.clf.loss_function(
                    reconstruction, batch, recons_features, input_features
                )
                loss = loss_dict["loss"]
                # Loss is a single scalar value.
                scores.append(loss.cpu().numpy())

        decision_scores = np.array(scores)
        # logging.info(
        #     f"Computed decision scores: min={decision_scores.min():.4f}, "
        #     f"max={decision_scores.max():.4f}, mean={decision_scores.mean():.4f}"
        # )
        return decision_scores

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Act chooses to either ask for help or not, which is equivalent to asking whther
        the input observation is normal or OOD.
        """
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
        scores: np.ndarray = self._compute_decision_scores(observation)

        # Use our own threshold instead of self.clf.threshold_
        action: np.ndarray = (scores > self.threshold_).astype(int)

        if not np.any(action == 0) and not np.any(action == 1):
            logging.warning("No action selected as normal or OOD")

        return action

    def update_params(self, params: Dict[str, Any]) -> None:
        """Override to update our threshold instead of params dict."""

        super().update_params(params)
        if "threshold" in params:
            self.threshold_ = params["threshold"]
        else:
            raise ValueError("Threshold not found in params")

    def save_model(self, name: str, save_dir: str) -> None:
        """
        Override parent save_model to save Lightning model and decision scores.
        """
        save_path: Path = Path(save_dir) / f"{name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and decision scores together
        save_dict = {
            "model": self.clf,
            "_train_decision_scores": self._train_decision_scores,
            "threshold_": self.threshold_,
            "clf_name": self.clf_name,
        }

        torch.save(save_dict, save_path)
        logging.info(f"Saved Lightning AE policy with decision scores to {save_path}")

    def load_model(self, load_dir: str) -> "LightningAEPolicy":
        """
        Override parent load_model to load Lightning model and decision scores.
        """
        load_path: Path = Path(load_dir)

        # Load the saved dictionary
        save_dict = torch.load(load_path, map_location=self.device)

        # Restore model
        self.clf = save_dict["model"]
        self.clf.to(self.device)

        # Restore decision scores and other attributes
        self._train_decision_scores = save_dict.get("_train_decision_scores", None)
        self.clf_name = save_dict.get("clf_name", "LightningAE")
        self.threshold_ = save_dict.get("threshold_", 0.0)

        logging.info(f"Loaded Lightning AE model with decision scores from {load_path}")
        logging.info(
            f"Restored decision scores: train_scores={'present' if self._train_decision_scores is not None else 'None'}"
        )

        return self

    def train_percentile(self, percentile: float) -> float:
        return np.percentile(self._train_decision_scores, percentile)

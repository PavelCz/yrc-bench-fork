import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import sys


from YRC.policies.ood import OODPolicy
from YRC.core.configs.global_configs import get_global_variable
from YRC.core.dataset import ObservationDataset, ObservationDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
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

    def __init__(self, config: Any, env: Any) -> None:
        # Initialize parent class
        super().__init__(config, env)

        # Lightning-specific attributes - use self.model instead of self.clf
        self.clf: Optional[BaseVAE] = None
        self.experiment: Optional[VAEXperiment] = None

        self.batch_size = None

        # OOD detection attributes
        self.threshold_: float = 0.0
        self.decision_scores_: Optional[np.ndarray] = None

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

        # Initialize model
        if method_name not in vae_models:
            raise ValueError(
                f"Model {method_name} not found in vae_models registry. "
                f"Available models: {list(vae_models.keys())}"
            )

        self.clf = vae_models[method_name](**model_config["model_params"])
        self.experiment = VAEXperiment(self.clf, model_config["exp_params"])

        self.experiment.to(self.device)

        save_dir = get_global_variable("experiment_dir")

        tb_logger = TensorBoardLogger(save_dir=save_dir, name=self.args.exp_name)
        
        if self.device.type == "cuda":
            accelerator = "auto"
        elif self.device.type == "cpu":
            accelerator = "cpu"
        else:
            raise ValueError(f"Invalid device type: {self.device.type}")

        self.runner = Trainer(
            logger=tb_logger,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(
                    save_top_k=1,
                    dirpath=Path(tb_logger.log_dir) / "checkpoints",
                    filename="best",
                    monitor="val_loss",
                    save_last=True,
                ),
            ],
            strategy=DDPPlugin(find_unused_parameters=False),
            max_epochs=epochs,
            accelerator=accelerator,
            # devices=self.device.index,
        )

        # Move to device
        self.clf.to(self.device)

        # Set clf_name for compatibility but use self.model instead of self.clf
        self.clf_name = "LightningAE"

        logging.info(f"Initialized Lightning AE model: {method_name}")
        logging.info(f"Input shape: {dummy_obs_shape}")
        logging.info(f"Latent dim: {model_config['model_params']['latent_dim']}")

    def decision_function(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Compute anomaly scores for input samples using the Lightning model."""
        if self.clf is None:
            raise ValueError(
                "Model not initialized. Call initialize_ood_detector first."
            )

        raise NotImplementedError("Decision function not implemented for Lightning AE")

        self.clf.eval()
        scores: List[float] = []

        # Ensure input is on the correct device and format
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        with torch.no_grad():
            batch_size: int = 32
            for i in range(0, len(x), batch_size):
                batch: torch.Tensor = x[i : i + batch_size]

                # Handle different input types
                if len(batch.shape) == 2:  # Flattened data
                    if self.feature_type == "obs":
                        # Reshape to image format
                        size: int = int(
                            np.sqrt(batch.shape[1] // self.model_config["in_channels"])
                        )
                        batch = batch.view(
                            -1, self.model_config["in_channels"], size, size
                        )

                # Get reconstruction
                if hasattr(self.clf, "forward"):
                    reconstruction: torch.Tensor = self.clf(batch)[0]
                else:
                    reconstruction = self.clf.decode(self.clf.encode(batch)[0])

                # Compute reconstruction error
                mse: torch.Tensor = F.mse_loss(reconstruction, batch, reduction="none")
                mse = mse.view(mse.shape[0], -1).mean(dim=1)
                scores.extend(mse.cpu().numpy())

        return np.array(scores)

    def fit(
        self, x: torch.Tensor, x_threshold: torch.Tensor, y: Optional[Any] = None
    ) -> None:
        """
        Override parent fit method to work with Lightning model instead of self.clf.
        """
        logging.info(
            "Lightning AE Policy: Computing decision scores for threshold setting"
        )


        x = x.to(self.device)
        x_threshold = x_threshold.to(self.device)

        # Turn sequence of observations x into a dataset
        train_dataset = ObservationDataset(x)
        test_dataset = ObservationDataset(x_threshold)

        datamodule = ObservationDataModule(
            **self.data_config,
            train_dataset_torch=train_dataset,
            test_dataset_torch=test_dataset,
            test_batch_size=self.batch_size,
        )

        self.runner.fit(self.experiment, datamodule=datamodule)

        # Run test run to generate samples. Uses test dataset from datamodule as
        # specified above.
        self.runner.test(self.experiment)

        # Compute decision scores for threshold setting
        # self.decision_scores_ = self._compute_decision_scores(x)

    def _compute_decision_scores(self, x: torch.Tensor) -> np.ndarray:
        """Compute reconstruction error scores on the training data."""
        if self.clf is None:
            raise ValueError("Model not initialized")

        raise NotImplementedError("Decision function not implemented for Lightning AE")

        self.clf.eval()
        scores: List[float] = []

        with torch.no_grad():
            batch_size: int = 32  # Process in batches to avoid memory issues
            for i in range(0, len(x), batch_size):
                batch: torch.Tensor = x[i : i + batch_size].to(self.device)

                # Handle different input types
                if len(batch.shape) == 2:  # Flattened data
                    # Reshape to appropriate image format if needed
                    if self.feature_type == "obs":
                        # Assume square images, infer dimensions
                        size: int = int(
                            np.sqrt(batch.shape[1] // self.model_config["in_channels"])
                        )
                        batch = batch.view(
                            -1, self.model_config["in_channels"], size, size
                        )

                # Get reconstruction
                if hasattr(self.clf, "forward"):
                    reconstruction: torch.Tensor = self.clf(batch)[
                        0
                    ]  # First output is reconstruction
                else:
                    reconstruction = self.clf.decode(self.clf.encode(batch)[0])

                # Compute reconstruction error (MSE)
                mse: torch.Tensor = F.mse_loss(reconstruction, batch, reduction="none")
                # Average over all dimensions except batch
                mse = mse.view(mse.shape[0], -1).mean(dim=1)
                scores.extend(mse.cpu().numpy())

        decision_scores = np.array(scores)
        logging.info(
            f"Computed decision scores: min={decision_scores.min():.4f}, "
            f"max={decision_scores.max():.4f}, mean={decision_scores.mean():.4f}"
        )
        return decision_scores

    def act(self, obs, greedy: bool = False) -> np.ndarray:
        """
        Override parent act method to use our decision_function instead of
        self.clf.decision_function.
        """
        raise NotImplementedError("Act method not implemented for Lightning AE")

        keys: List[str] = {
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
                self.to_tensor(obs[key]["image"] if key == "env_obs" else obs[key])
                for key in keys
            ]
        else:
            observation = [self.to_tensor(obs[key]) for key in keys]

        if self.feature_type in ["obs", "hidden", "dist"]:
            observation = observation[0]

        # For autoencoder, we might need to flatten or concatenate features
        if isinstance(observation, list):
            # Concatenate features along the last dimension
            observation = torch.cat(
                [o.flatten(start_dim=1) for o in observation], dim=-1
            )

        # Flatten for autoencoder input if needed
        if len(observation.shape) > 2:
            observation = observation.flatten(start_dim=1)

        # Use our own decision_function instead of self.clf.decision_function
        score: np.ndarray = self.decision_function(observation)

        # Use our own threshold instead of self.clf.threshold_
        action: np.ndarray = (score > self.threshold_).astype(int)

        if not np.any(action == 0) and not np.any(action == 1):
            logging.warning("No action selected as normal or OOD")

        return action

    def update_params(self, params: Dict[str, Any]) -> None:
        """Override to update our threshold instead of params dict."""
        raise NotImplementedError(
            "Update params method not implemented for Lightning AE"
        )

        super().update_params(params)
        if "threshold" in params:
            self.threshold_ = params["threshold"]

    def save_model(self, name: str, save_dir: str) -> None:
        """
        Override parent save_model to save Lightning model instead of self.clf.
        """
        save_path: Path = Path(save_dir) / f"{name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.clf, save_path)

        # Save Lightning checkpoint separately
        # ckpt_path: str = Path(save_dir) / f"{name}_lightning.ckpt"
        # if self.experiment is not None and hasattr(self.experiment, "trainer"):
        #     self.experiment.trainer.save_checkpoint(ckpt_path)

        # Save policy state (compatible with parent class structure)
        # state_dict: Dict[str, Any] = {
        #     "clf": self.clf,
        #     "class_name": self.__class__.__name__,
        #     "clf_name": self.clf_name,
        #     # "model_config": self.model_config,
        #     # "exp_config": self.exp_config,
        #     # "params": self.params,
        #     # "threshold_": self.threshold_,
        #     # "decision_scores_": self.decision_scores_,
        #     # "contamination": self.contamination,
        #     # "feature_type": self.feature_type,
        #     # "lightning_ckpt_path": ckpt_path,
        # }

        # from joblib import dump

        # dump(state_dict, save_path)
        # logging.info(f"Saved Lightning AE policy to {save_path}")

    def load_model(self, load_dir: str) -> "LightningAEPolicy":
        """
        Override parent load_model to load Lightning model instead of self.clf.
        """
        load_path: Path = Path(load_dir)
        self.clf = torch.load(load_path)
        self.clf.to(self.device)
        self.clf_name = "LightningAE"
        logging.info(f"Loaded Lightning AE model from {load_path}")
        return self

        # load_dir = Path(load_dir)

        # if load_dir.endswith(".joblib"):
        #     state_dict: Dict[str, Any] = load(load_dir)
        # else:
        #     state_dict = load(load_dir / "model.joblib")

        # self.model_config = state_dict["model_config"]
        # self.exp_config = state_dict["exp_config"]
        # self.params = state_dict["params"]
        # self.threshold_ = state_dict.get("threshold_", 0.0)
        # self.decision_scores_ = state_dict.get("decision_scores_")
        # self.contamination = state_dict.get("contamination", 0.1)
        # self.feature_type = state_dict["feature_type"]
        # self.clf_name = state_dict.get("clf_name", "LightningAE")

        # # Reinitialize Lightning model
        # self.clf = vae_models[self.model_config["name"]](**self.model_config)
        # self.experiment = VAEXperiment(self.clf, self.exp_config)

        # # Load Lightning checkpoint if available
        # ckpt_path: Optional[str] = state_dict.get("lightning_ckpt_path")
        # if ckpt_path and os.path.exists(ckpt_path):
        #     self.experiment = VAEXperiment.load_from_checkpoint(
        #         ckpt_path, vae_model=self.clf, params=self.exp_config
        #     )
        #     self.clf = self.experiment.model

        # self.clf.to(self.device)

        # return self

    def to_tensor(
        self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any], tuple]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], tuple]:
        """Converts input to a torch tensor if it's not already."""
        if isinstance(data, dict):
            for key in data:
                data[key] = self.to_tensor(data[key])
            return data
        if isinstance(data, tuple):
            return data
        if not torch.is_tensor(data):
            return torch.from_numpy(data).float().to(self.device)
        return data

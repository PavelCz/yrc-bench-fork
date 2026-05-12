# -*- coding: utf-8 -*-
"""Deep One-Class Classification for outlier detection"""
# Author: Rafal Bodziony <bodziony.rafal@gmail.com> for the TensorFlow version
# Author: Yuehan Qin <yuehanqi@usc.edu> for the PyTorch version
# License: BSD 2 clause

import numpy as np

try:
    import torch
except ImportError:
    print("please install torch first")

import torch
import torch.nn as nn
import torch.optim as optim

import logging

from torch.utils.data import DataLoader, Dataset, TensorDataset

from .base import BaseDetector
from ..utils.torch_utility import get_activation_by_name
from ..utils.utility import check_parameter

optimizer_dict = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adamw": optim.AdamW,
    "nadam": optim.NAdam,
    "sparseadam": optim.SparseAdam,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
}

PROGRESS_LOG_BATCH_INTERVAL = 100


class InnerDeepSVDD(nn.Module):
    """Inner class for DeepSVDD model.

    Parameters
    ----------
    n_features:
        Number of features in the input data.

    use_ae: bool, optional (default=False)
        The AutoEncoder type of DeepSVDD it reverse neurons from hidden_neurons
        if set to True.

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. if use_ae is True, neurons
        will be reversed eg. [64, 32] -> [64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
    """

    def __init__(
        self,
        n_features,
        use_ae,
        hidden_neurons,
        hidden_activation,
        output_activation,
        dropout_rate,
        l2_regularizer,
        feature_type,
        benchmark,
        input_shape=None,
        center_init_post_activation=False,
    ):
        super(InnerDeepSVDD, self).__init__()
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.feature_type = feature_type
        self.benchmark = benchmark
        self.input_shape = input_shape
        self.center_init_post_activation = center_init_post_activation

        if self.feature_type in ["obs", "obs_dist", "hidden_obs", "obs_hidden_dist"]:
            self.embedder_features = n_features
            self.embedder = self._build_embedder()

        if self.feature_type == "obs":
            self.linear_features = n_features
        elif self.feature_type in ["hidden", "dist"]:
            self.linear_features = self.input_shape[1]
        elif self.feature_type in ["hidden_obs", "obs_dist"]:
            self.linear_features = n_features + self.input_shape[-1]
        elif self.feature_type == "hidden_dist":
            self.linear_features = self.input_shape[1] + self.input_shape[2]
        elif self.feature_type == "obs_hidden_dist":
            self.linear_features = (
                n_features + self.input_shape[-2] + self.input_shape[-1]
            )

        self.fc_part = self._build_fc()
        self.c = None  # Center of the hypersphere for DeepSVDD

    def _init_c(self, X_norm, eps=0.1):
        intermediate_output = {}
        # When center_init_post_activation is True, the center c is captured
        # from image(phi) — i.e. after the trailing activation — as required
        # by Ruff et al. (2018) §3.1, §4.1. The default (False) reproduces
        # the unpatched modanesh/pyod behaviour. See
        # docs/image_svdd_collapse_bugs.md, Bug 2.
        hook_target_name = (
            f"hidden_activation_e{len(self.hidden_neurons)}"
            if self.center_init_post_activation
            else "net_output"
        )
        hook_handle = self.fc_part._modules.get(hook_target_name).register_forward_hook(
            lambda module, input, output: intermediate_output.update(
                {"net_output": output}
            )
        )
        if self.feature_type in ["obs", "hidden"]:
            self.forward(X_norm)
        elif self.feature_type == "dist":
            X_norm = X_norm.softmax(dim=-1)
            self.forward(X_norm)
        elif self.feature_type in ["hidden_obs", "hidden_dist", "obs_dist"]:
            self.forward([X_norm[0], X_norm[1]])
        elif self.feature_type in ["obs_hidden_dist"]:
            self.forward([X_norm[0], X_norm[1], X_norm[2]])
        out = intermediate_output["net_output"]
        hook_handle.remove()
        self.c = torch.mean(out, dim=0)
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps

    def _build_embedder(self):
        if len(self.input_shape) == 3:
            channels = self.input_shape[0]
        else:
            channels = self.input_shape[1]
        layers = nn.Sequential()
        # bias=False on every encoder layer is a Deep SVDD requirement
        # (Ruff et al. 2018, Proposition 2 / §4.1); see
        # docs/image_svdd_collapse_bugs.md.
        layers.add_module(
            "cnn_layer1",
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
        )
        layers.add_module("cnn_activation1", nn.ReLU())
        layers.add_module("cnn_pool", nn.MaxPool2d(kernel_size=2, stride=2))
        layers.add_module(
            "cnn_layer2",
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
        )
        layers.add_module("cnn_activation2", nn.ReLU())
        layers.add_module("cnn_adaptive_pool", nn.AdaptiveMaxPool2d((32, 32)))
        layers.add_module("flatten", nn.Flatten())
        layers.add_module(
            "cnn_fc", nn.Linear(32 * 32 * 32, self.embedder_features, bias=False)
        )
        layers.add_module("cnn_fc_activation", nn.ReLU())
        return layers

    def _build_fc(self):
        layers = nn.Sequential()
        layers.add_module(
            "input_layer",
            nn.Linear(self.linear_features, self.hidden_neurons[0], bias=False),
        )
        layers.add_module(
            "hidden_activation_e0", get_activation_by_name(self.hidden_activation)
        )
        for i in range(1, len(self.hidden_neurons) - 1):
            layers.add_module(
                f"hidden_layer_e{i}",
                nn.Linear(
                    self.hidden_neurons[i - 1], self.hidden_neurons[i], bias=False
                ),
            )
            layers.add_module(
                f"hidden_activation_e{i}",
                get_activation_by_name(self.hidden_activation),
            )
            layers.add_module(f"hidden_dropout_e{i}", nn.Dropout(self.dropout_rate))
        layers.add_module(
            "net_output",
            nn.Linear(self.hidden_neurons[-2], self.hidden_neurons[-1], bias=False),
        )
        layers.add_module(
            f"hidden_activation_e{len(self.hidden_neurons)}",
            get_activation_by_name(self.hidden_activation),
        )

        if self.use_ae:
            # Add reverse layers for the autoencoder if needed
            for j in range(len(self.hidden_neurons) - 1, 0, -1):
                layers.add_module(
                    f"hidden_layer_d{j}",
                    nn.Linear(
                        self.hidden_neurons[j], self.hidden_neurons[j - 1], bias=False
                    ),
                )
                layers.add_module(
                    f"hidden_activation_d{j}",
                    get_activation_by_name(self.hidden_activation),
                )
                layers.add_module(f"hidden_dropout_d{j}", nn.Dropout(self.dropout_rate))
            layers.add_module(
                "output_layer",
                nn.Linear(self.hidden_neurons[0], self.n_features, bias=False),
            )
            layers.add_module(
                "output_activation", get_activation_by_name(self.output_activation)
            )

        return layers

    def forward(self, x):
        feature_type_to_processing = {
            "obs": lambda x: self.embedder(x),
            "hidden": lambda x: x,
            "hidden_obs": lambda x: torch.cat([self.embedder(x[0]), x[1]], dim=-1),
            "dist": lambda x: x.softmax(dim=-1),
            "hidden_dist": lambda x: torch.cat([x[0], x[1].softmax(dim=-1)], dim=-1),
            "obs_dist": lambda x: torch.cat(
                [self.embedder(x[0]), x[1].softmax(dim=-1)], dim=-1
            ),
            "obs_hidden_dist": lambda x: torch.cat(
                [self.embedder(x[0]), x[1], x[2].softmax(dim=-1)], dim=-1
            ),
        }
        x = feature_type_to_processing[self.feature_type](x)
        x = self.fc_part(x)
        return x


class DeepSVDD(BaseDetector):
    """Deep One-Class Classifier with AutoEncoder (AE) is a type of neural
    networks for learning useful data representations in an unsupervised way.
    DeepSVDD trains a neural network while minimizing the volume of a
    hypersphere that encloses the network representations of the data,
    forcing the network to extract the common factors of variation.
    Similar to PCA, DeepSVDD could be used to detect outlying objects in the
    data by calculating the distance from center
    See :cite:`ruff2018deepsvdd` for details.

    Parameters
    ----------
    n_features: int,
        Number of features in the input data.

    c: float, optional (default='forwad_nn_pass')
        Deep SVDD center, the default will be calculated based on network
        initialization first forward pass. To get repeated results set
        random_state if c is set to None.

    use_ae: bool, optional (default=False)
        The AutoEncoder type of DeepSVDD it reverse neurons from hidden_neurons
        if set to True.

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. if use_ae is True, neurons
        will be reversed eg. [64, 32] -> [64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(
        self,
        n_features,
        c=None,
        use_ae=False,
        hidden_neurons=None,
        hidden_activation="relu",
        output_activation="sigmoid",
        optimizer="adam",
        epochs=100,
        batch_size=32,
        dropout_rate=0.2,
        l2_regularizer=0.1,
        feature_type="obs",
        benchmark="procgen",
        validation_size=0.1,
        preprocessing=True,
        verbose=1,
        random_state=None,
        contamination=0.1,
        input_shape=None,
        logger=None,
        explicit_wd_coef=1.0,
        center_init_post_activation=False,
    ):
        super(DeepSVDD, self).__init__(contamination=contamination)

        self.n_features = n_features
        self.c = c
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.feature_type = feature_type
        self.benchmark = benchmark
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None
        self.best_model_dict = None
        self.input_shape = input_shape
        self.logger = logger  # Wandb logger for training metrics
        # Coefficient on the explicit Frobenius w_d term in _loss.
        # Defaults to 1.0 (modanesh/pyod behaviour); set to 0.0 to drop the
        # term entirely. See docs/image_svdd_collapse_bugs.md, Bug 3.
        self.explicit_wd_coef = explicit_wd_coef
        # See InnerDeepSVDD.__init__.
        self.center_init_post_activation = center_init_post_activation

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        check_parameter(
            dropout_rate, 0, 1, param_name="dropout_rate", include_left=True
        )

        # Initialize the DeepSVDD model with updated input shape
        self.model_ = InnerDeepSVDD(
            n_features=self.n_features,  # Now determined by CNN output
            use_ae=self.use_ae,
            hidden_neurons=self.hidden_neurons,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            dropout_rate=self.dropout_rate,
            l2_regularizer=self.l2_regularizer,
            feature_type=self.feature_type,
            benchmark=self.benchmark,
            input_shape=self.input_shape,
            center_init_post_activation=self.center_init_post_activation,
        )

    def fit(self, X, X_threshold, y=None, X_val=None, batch_transform=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : list or numpy array of shape (n_samples, channels, height, width)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        is_streaming = isinstance(X, Dataset)
        if is_streaming and self.feature_type not in ["obs", "hidden"]:
            raise ValueError(
                "Streaming DeepSVDD fit currently supports feature_type='obs' and "
                f"feature_type='hidden', got {self.feature_type!r}."
            )

        if self.benchmark == "minigrid" and not is_streaming:
            if self.feature_type == "obs_hidden_dist":
                # removing direction from the input. in minigrid, it's the "direction"
                X.pop(0)
                X_threshold.pop(0)
                # removing text from the input. in minigrid, it's the "mission"
                X.pop(1)
                X_threshold.pop(1)

        X_norm = X if is_streaming else self.normalization(X)
        X_norm_th = X_threshold if is_streaming else self.normalization(X_threshold)
        X_norm_val = self.normalization(X_val) if X_val is not None else None

        model_device = next(self.model_.parameters()).device

        if self.c is None:
            if is_streaming:
                center_dataloader = DataLoader(
                    X_norm,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                self._init_c_from_dataloader(
                    center_dataloader,
                    model_device,
                    batch_transform=batch_transform,
                )
            else:
                # We move things around devices a bit to avoid having to move the
                # entire dataset to the GPU.
                self.model_.to(X_norm.device)
                self.model_._init_c(X_norm)
                self.c = self.model_.c.detach().to(model_device)
                self.model_.to(model_device)

        # Prepare DataLoader for batch processing
        if is_streaming:
            dataloader = DataLoader(
                X_norm,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
        else:
            dataset = self._make_dataset(X_norm)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = None
        if X_norm_val is not None:
            val_dataset = self._make_dataset(X_norm_val)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        best_loss = float("inf")
        best_val_loss = float("inf")
        best_model_dict = None
        optimizer = optimizer_dict[self.optimizer](
            self.model_.parameters(), weight_decay=self.l2_regularizer
        )

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0
            num_batches = len(dataloader)
            self._log_epoch_start(epoch, num_batches, is_streaming=is_streaming)
            for batch_index, batch in enumerate(dataloader, start=1):
                if is_streaming:
                    batch_x = self._streaming_batch_to_input(
                        batch, model_device, batch_transform=batch_transform
                    )
                else:
                    batch_x = self._batch_to_input(batch, model_device)
                outputs = self.model_(batch_x)
                loss = self._loss(outputs, batch_x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                running_loss = epoch_loss / batch_index
                if self._should_log_progress(batch_index, num_batches):
                    self._log_training_progress(
                        epoch,
                        batch_index,
                        num_batches,
                        batch_loss=loss.item(),
                        running_loss=running_loss,
                    )
            epoch_loss /= len(dataloader)
            val_loss = None
            if val_dataloader is not None:
                val_loss = self._evaluate_loss(val_dataloader, model_device)

            log_message = f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss}"
            if val_loss is not None:
                log_message += f", Val Loss: {val_loss}"
            logging.info(log_message)

            # Log to wandb if logger is available
            if self.logger is not None:
                metrics = {
                    "train/loss": epoch_loss,
                    "train/epoch": epoch + 1,
                    "train/best_loss": best_loss
                    if epoch_loss >= best_loss
                    else epoch_loss,
                }
                if val_loss is not None:
                    metrics["val/loss"] = val_loss
                    metrics["val/best_loss"] = (
                        best_val_loss if val_loss >= best_val_loss else val_loss
                    )
                self.logger.log_metrics(metrics, step=epoch + 1)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_dict = self.model_.state_dict()
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
        self.best_model_dict = best_model_dict

        if is_streaming:
            scores = self._streaming_decision_scores(
                X_norm_th,
                model_device,
                batch_transform=batch_transform,
            )
        else:
            scores = []
            for x_i in X_norm_th:
                x_i = x_i.to(model_device)
                scores.append(self.decision_function(x_i.unsqueeze(0)))
        self.decision_scores_ = np.concatenate(scores, axis=0)
        self._process_decision_scores()
        return self

    def _should_log_progress(self, batch_index, num_batches):
        return (
            batch_index == 1
            or batch_index == num_batches
            or batch_index % PROGRESS_LOG_BATCH_INTERVAL == 0
        )

    def _log_progress_message(self, message):
        logging.info(message)
        print(message, flush=True)

    def _log_epoch_start(self, epoch, num_batches, is_streaming):
        message = (
            f"Starting DeepSVDD epoch {epoch + 1}/{self.epochs} "
            f"({num_batches} batches, streaming={is_streaming})"
        )
        self._log_progress_message(message)
        if self.logger is not None:
            self.logger.log_metrics(
                {
                    "train/epoch": epoch + 1,
                    "train/num_batches": num_batches,
                    "train/epoch_started": 1,
                    "train/streaming": int(is_streaming),
                }
            )

    def _log_training_progress(
        self, epoch, batch_index, num_batches, batch_loss, running_loss
    ):
        message = (
            f"DeepSVDD epoch {epoch + 1}/{self.epochs} "
            f"batch {batch_index}/{num_batches}: "
            f"batch_loss={batch_loss:.6g}, running_loss={running_loss:.6g}"
        )
        self._log_progress_message(message)
        if self.logger is not None:
            self.logger.log_metrics(
                {
                    "train/epoch": epoch + 1,
                    "train/batch": batch_index,
                    "train/num_batches": num_batches,
                    "train/epoch_progress": batch_index / num_batches,
                    "train/batch_loss": batch_loss,
                    "train/running_loss": running_loss,
                }
            )

    def _make_dataset(self, X_norm):
        if self.feature_type in [
            "hidden_obs",
            "hidden_dist",
            "obs_dist",
            "obs_hidden_dist",
        ]:
            return TensorDataset(*X_norm, *X_norm)
        return TensorDataset(X_norm, X_norm)

    def _batch_to_input(self, batch, model_device):
        if torch.is_tensor(batch):
            return batch.to(model_device)
        batch = [b.to(model_device) for b in batch]
        if self.feature_type in ["hidden_obs", "hidden_dist", "obs_dist"]:
            return batch[0], batch[1]
        if self.feature_type in ["obs_hidden_dist"]:
            return batch[0], batch[1], batch[2]
        return batch[0]

    def _streaming_batch_to_input(self, batch, model_device, batch_transform=None):
        if batch_transform is not None:
            batch = batch_transform(batch)
        batch = self._unwrap_single_tensor_batch(batch)
        batch = self.normalization(batch)
        return batch.to(model_device)

    def _unwrap_single_tensor_batch(self, batch):
        if torch.is_tensor(batch):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            return batch[0]
        raise ValueError(
            f"Expected streaming DeepSVDD batch to be a tensor, got {type(batch)}."
        )

    def _init_c_from_dataloader(
        self, dataloader, model_device, batch_transform=None, eps=0.1
    ):
        intermediate_output = {}
        # Mirrors InnerDeepSVDD._init_c: capture c from image(phi) iff
        # center_init_post_activation is set. See
        # docs/image_svdd_collapse_bugs.md, Bug 2.
        hook_target_name = (
            f"hidden_activation_e{len(self.hidden_neurons)}"
            if self.center_init_post_activation
            else "net_output"
        )
        hook_handle = self.model_.fc_part._modules.get(
            hook_target_name
        ).register_forward_hook(
            lambda module, input, output: intermediate_output.update(
                {"net_output": output}
            )
        )

        output_sum = None
        output_count = 0
        self.model_.eval()
        num_batches = len(dataloader)
        self._log_progress_message(
            f"Initializing DeepSVDD center ({num_batches} batches)"
        )
        if self.logger is not None:
            self.logger.log_metrics(
                {
                    "train/center_init_started": 1,
                    "train/center_init_num_batches": num_batches,
                }
            )
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader, start=1):
                batch_x = self._streaming_batch_to_input(
                    batch, model_device, batch_transform=batch_transform
                )
                self.model_(batch_x)
                out = intermediate_output["net_output"]
                output_sum = (
                    out.sum(dim=0)
                    if output_sum is None
                    else output_sum + out.sum(dim=0)
                )
                output_count += out.shape[0]
                if self._should_log_progress(batch_index, num_batches):
                    self._log_center_init_progress(batch_index, num_batches)
        hook_handle.remove()

        if output_count == 0:
            raise ValueError("Cannot initialize DeepSVDD center from an empty dataset.")

        self.c = (output_sum / output_count).detach()
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps
        self.model_.c = self.c
        self.model_.train()

    def _log_center_init_progress(self, batch_index, num_batches):
        message = (
            "DeepSVDD center init "
            f"batch {batch_index}/{num_batches} "
            f"({batch_index / num_batches:.1%})"
        )
        self._log_progress_message(message)
        if self.logger is not None:
            self.logger.log_metrics(
                {
                    "train/center_init_batch": batch_index,
                    "train/center_init_num_batches": num_batches,
                    "train/center_init_progress": batch_index / num_batches,
                }
            )

    def _streaming_decision_scores(self, dataset, model_device, batch_transform=None):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        scores = []
        self.model_.eval()
        num_batches = len(dataloader)
        self._log_progress_message(
            f"Computing DeepSVDD training decision scores ({num_batches} batches)"
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader, start=1):
                batch_x = self._streaming_batch_to_input(
                    batch, model_device, batch_transform=batch_transform
                )
                outputs = self.model_(batch_x)
                dist = torch.sum((outputs - self.c) ** 2, dim=-1)
                scores.append(dist.cpu().numpy())
                if self._should_log_progress(batch_index, num_batches):
                    self._log_score_progress(batch_index, num_batches)
        return scores

    def _log_score_progress(self, batch_index, num_batches):
        message = (
            "DeepSVDD decision-score pass "
            f"batch {batch_index}/{num_batches} "
            f"({batch_index / num_batches:.1%})"
        )
        self._log_progress_message(message)
        if self.logger is not None:
            self.logger.log_metrics(
                {
                    "train/decision_score_batch": batch_index,
                    "train/decision_score_num_batches": num_batches,
                    "train/decision_score_progress": batch_index / num_batches,
                }
            )

    def _loss(self, outputs, batch_x):
        dist = torch.sum((outputs - self.c) ** 2, dim=-1)
        # explicit_wd_coef defaults to 1.0 (modanesh/pyod behaviour). Set to
        # 0.0 to drop the explicit regulariser and rely solely on the
        # optimiser's weight_decay (upstream yzhao062/pyod uses 1e-6 here,
        # which is functionally equivalent to 0.0). See
        # docs/image_svdd_collapse_bugs.md, Bug 3.
        w_d = self.explicit_wd_coef * sum(
            [torch.linalg.norm(w) for w in self.model_.parameters()]
        )

        if self.use_ae:
            return torch.mean(dist) + w_d + torch.mean(torch.square(outputs - batch_x))
        return torch.mean(dist) + w_d

    def _evaluate_loss(self, dataloader, model_device):
        self.model_.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_x = self._batch_to_input(batch, model_device)
                outputs = self.model_(batch_x)
                total_loss += self._loss(outputs, batch_x).item()
        self.model_.train()
        return total_loss / len(dataloader)

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on the DeepSVDD model.
        Outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, channels, height, width)
            The input samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        # Normalize data if pixel values are in [0, 255] range
        if isinstance(X[0], dict):
            X[0] = X[0]["image"]
        X = self.normalization(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X)
            dist = torch.sum((outputs - self.c) ** 2, dim=-1)
        anomaly_scores = dist.cpu().numpy()
        return anomaly_scores

    def normalization(self, X):
        if self.feature_type in ["obs", "hidden_obs", "obs_dist", "obs_hidden_dist"]:
            X_img = X if self.feature_type == "obs" else X[0]
            # Normalize the image data if pixel values are in the range [0, 255]
            if X_img.max() > 1:
                X_img = X_img / 255.0
            X_norm = X_img if self.feature_type == "obs" else [X_img, *X[1:]]
        elif self.feature_type in ["hidden", "dist", "hidden_dist"]:
            X_norm = X
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        return X_norm

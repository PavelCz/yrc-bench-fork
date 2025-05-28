import math
import importlib
import torch
import torch.nn as nn
import logging
from typing import List
import numpy as np

if importlib.util.find_spec("gymnasium") is None:
    import gym
else:
    import gymnasium as gym
import numpy as np
import re

# Import the AutoEncoder base class
from lib.pyod.pyod.models.auto_encoder import AutoEncoder
from torch.utils.data import DataLoader
from sklearn.utils import check_array
from lib.pyod.pyod.utils.torch_utility import TorchDataset


def orthogonal_init(module, gain=nn.init.calculate_gain("relu")):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class ImpalaModel(nn.Module):
    def __init__(self, input_size, scale=1):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=input_size[0], out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)

        fc_input_size = self._get_fc_input_size(input_size)

        self.fc = nn.Linear(in_features=fc_input_size, out_features=256)
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def _get_fc_input_size(self, input_size):
        test_in = torch.zeros((1,) + input_size)
        test_out = self.block3(self.block2(self.block1(test_in)))
        return math.prod(test_out.shape[1:])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        if torch.isnan(x).any():
            print("ImpalaModel output shape:", x.shape)
            print("ImpalaModel output contains NaN:", torch.isnan(x).any())
        return x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to np array
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return DictList({
                "image": preprocess_images(obss['image'], device=device),
                "text": preprocess_texts(obss["mission"], vocab, device=device)
            })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


class AutoEncoderWithVal(AutoEncoder):
    """
    AutoEncoder with validation functionality built-in.
    
    This subclass extends AutoEncoder to include validation during training.
    It automatically evaluates the model on a validation dataset after each epoch
    and stores the validation scores for later analysis.
    
    Additional Parameters
    ----------
    validation_loader : torch.utils.data.DataLoader, optional (default=None)
        The data loader for validation data. Can be set later using set_validation_loader().
        
    score_list : List[np.ndarray], optional (default=None)
        List to store validation scores from each epoch. If None, a new list will be created.
    """
    
    def __init__(
            self,
            contamination=0.1, preprocessing=True,
            lr=1e-3, epoch_num=10, batch_size=32,
            optimizer_name='adam',
            device=None, random_state=42,
            use_compile=False, compile_mode='default',
            verbose=1,
            optimizer_params: dict = {'weight_decay': 1e-5},
            hidden_neuron_list=[64, 32],
            hidden_activation_name='relu',
            batch_norm=True, dropout_rate=0.2,
            training_loader: DataLoader = None,
        ):
        
        super(AutoEncoderWithVal, self).__init__(
            contamination=contamination,
            preprocessing=preprocessing,
            lr=lr, epoch_num=epoch_num, batch_size=batch_size,
            optimizer_name=optimizer_name,
            device=device, random_state=random_state,
            use_compile=use_compile, compile_mode=compile_mode,
            verbose=verbose,
            optimizer_params=optimizer_params,
            hidden_neuron_list=hidden_neuron_list,
            hidden_activation_name=hidden_activation_name,
            batch_norm=batch_norm, dropout_rate=dropout_rate)
        
        self._validation_loader = None
        self.val_score_list = []
        self.training_loader = training_loader
        self.training_score_list = []
        
    def set_loaders(self, train_dataset: np.ndarray, val_dataset: np.ndarray):
        """Set the validation data loader."""
        train_dataset = check_array(train_dataset)
        val_dataset = check_array(val_dataset)

        if self.preprocessing:
            X_mean = np.mean(train_dataset, axis=0)
            X_std = np.std(train_dataset, axis=0)
            train_set = TorchDataset(X=train_dataset, y=None,
                                     mean=X_mean, std=X_std)
            X_mean = np.mean(val_dataset, axis=0)
            X_std = np.std(val_dataset, axis=0)
            val_set = TorchDataset(X=val_dataset, y=None,
                                     mean=X_mean, std=X_std)
        else:
            train_set = TorchDataset(X=train_dataset, y=None)
            val_set = TorchDataset(X=val_dataset, y=None)

        self.training_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        self._validation_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False
        )
        
    def epoch_update(self):
        """
        Called after each training epoch.
        Runs validation if validation loader is available and stores the scores.
        """
        # Call parent's epoch_update in case it has any functionality
        super().epoch_update()
        
        if self._validation_loader is None:
            logging.warning(
                "No validation loader set for OOD detector. Skipping validation."
            )
            return
        # Evaluate on validation data
        val_scores = self.evaluate(self._validation_loader)
        # Save the scores to the score list for later analysis
        self.val_score_list.append(val_scores)
        training_scores = self.evaluate(self.training_loader)
        self.training_score_list.append(training_scores)
        
        # Since evaluate() sets the model to eval mode, we need to set it back to train 
        # mode.
        self.model.train()

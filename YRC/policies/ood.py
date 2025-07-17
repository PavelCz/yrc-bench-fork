import os
import numpy as np
from copy import deepcopy as dc
import logging
from YRC.core import Policy
from lib.pyod.pyod.models import deep_svdd
from joblib import dump, load
from YRC.core.configs.global_configs import get_global_variable
from YRC.models.utils import AutoEncoderWithVal
from YRC.core.utils import to_tensor



class OODPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        if config.coord_policy.collect_data_agent == "weak":
            self.agent = env.weak_agent
        elif config.coord_policy.collect_data_agent == "strong":
            self.agent = env.strong_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0}
        self.clf = None
        self.clf_name = None
        self.device = get_global_variable("device")
        self.feature_type = config.coord_policy.feature_type


    def update_params(self, params):
        self.params = dc(params)
        if "threshold" not in params:
            raise ValueError(
                "Threshold is not in the provided params. "
                "You're probably doing something wrong"
            )
        self.clf.threshold_ = params["threshold"]

    def fit(self, x, x_threshold, y=None):
        if self.clf_name == "DeepSVDD":
            x = x.to(self.device)
            x_threshold = x_threshold.to(self.device)
            self.clf.fit(X=x, X_threshold=x_threshold, y=y)
        elif self.clf_name == "AutoEncoder":
            x = x.cpu()
            # Flatten the observations.
            x = x.reshape(x.shape[0], -1)

            x_threshold = x_threshold.cpu()
            x_threshold = x_threshold.reshape(x_threshold.shape[0], -1)

            self.clf.set_loaders(x, x_threshold)
            self.clf.fit(x, y)
        else:
            raise ValueError(f"Unknown OOD detector type: {self.clf_name}")

    def act(self, obs, greedy=False):
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
            observation = [to_tensor(obs[key]["image"] if key == "env_obs" else to_tensor(obs[key])) for key in keys]
        else:
            observation = [to_tensor(obs[key]) for key in keys]

        if self.feature_type in ["obs", "hidden", "dist"]:
            observation = observation[0]

        if self.clf_name == "AutoEncoder":
            # Since the AutoEncoder uses the default pyod implementation, it needs
            # tensors that can be converted to numpy arrays.
            observation = observation.cpu()
            # Additionally, the AutoEncoder expects a 2D array, so we flatten it.
            observation = observation.reshape(observation.shape[0], -1)
        elif self.clf_name == "DeepSVDD":
            observation = observation.to(self.device)

        score = self.clf.decision_function(observation)

        action = 1 - (score < self.clf.threshold_).astype(int)
        if 0 not in action and 1 not in action:
            print("No action is selected as OOD")
        return action

    def initialize_ood_detector(self, args, env):
        dummy_obs = env.reset()
        feature_type_to_shapes = {
            "obs": lambda dummy_obs: (
                dummy_obs['env_obs']['image'] if get_global_variable("benchmark") in ["cliport", "minigrid"] else
                dummy_obs['env_obs']
            ).shape,
            "hidden": lambda dummy_obs: dummy_obs['weak_features'].shape,
            "hidden_obs": lambda dummy_obs: (
                    (
                        dummy_obs['env_obs']['image'] if get_global_variable("benchmark") in ["cliport", "minigrid"] else dummy_obs['env_obs']
                    ).shape + dummy_obs['weak_features'].shape[1:]
            ),
            "dist": lambda dummy_obs: dummy_obs['weak_logit'].shape,
            "hidden_dist": lambda dummy_obs: (
                    dummy_obs['weak_features'].shape + dummy_obs['weak_logit'].shape[1:]
            ),
            "obs_dist": lambda dummy_obs: (
                    (
                        dummy_obs['env_obs']['image'] if get_global_variable("benchmark") in ["cliport", "minigrid"] else dummy_obs['env_obs']
                    ).shape + dummy_obs['weak_logit'].shape[1:]
            ),
            "obs_hidden_dist": lambda dummy_obs: (
                    (
                        dummy_obs['env_obs']['image'] if get_global_variable("benchmark") in ["cliport", "minigrid"] else dummy_obs['env_obs']
                    ).shape + dummy_obs['weak_features'].shape[1:] + dummy_obs['weak_logit'].shape[1:]
            ),
        }

        dummy_obs_shape = feature_type_to_shapes[self.feature_type](dummy_obs)

        if self.args.method == "DeepSVDD":
            self.clf_name = 'DeepSVDD'
            self.clf = deep_svdd.DeepSVDD(
                n_features=args.feature_size,
                use_ae=args.use_ae,
                contamination=args.contamination,
                epochs=args.epoch,
                batch_size=args.batch_size,
                input_shape=dummy_obs_shape,
                feature_type=self.feature_type,
                benchmark=get_global_variable("benchmark"),
            )
            self.clf.model_.to(self.device)
        elif self.args.method == "AutoEncoder":
            self.clf_name = 'AutoEncoder'
            clf = AutoEncoderWithVal(
                contamination=args.contamination,
                epoch_num=args.epoch,
                batch_size=args.batch_size,
                device=self.device,
                preprocessing=False,
            )
            self.clf = clf
        else:
            raise ValueError(f"Unknown OOD detector type: {args.ood_detector}")

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.joblib")
        state_dict = {
            'clf': self.clf,
            'class_name': self.__class__.__name__,
            'config': {
                'contamination': self.clf.contamination,
            },
            'clf_name': self.clf_name,
        }
        if type(self.clf) == deep_svdd.DeepSVDD:
            state_dict['config']['use_ae'] = self.clf.use_ae
        dump(state_dict, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_dir):
        state_dict = load(f"{load_dir}")
        self.clf = state_dict['clf']
        self.clf_name = state_dict['clf_name']

        return self

    def train_percentile(
        self, percentile: float
    ) -> float:
        return np.percentile(self.clf.decision_scores_, percentile)
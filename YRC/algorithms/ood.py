import numpy as np
import logging
from YRC.core import Algorithm
from YRC.core.configs.global_configs import get_global_variable
from typing import List
import torch
from torch.utils.data import Dataset


class OODAlgorithm(Algorithm):
    def __init__(self, config, env):
        super().__init__()
        self.args = config
        self.env = env
        self.save_dir = get_global_variable("experiment_dir")

    def train(
        self,
        policy,
        envs,
        rollout_obs: List[torch.Tensor],
        val_rollout_obs: List[torch.Tensor] = None,
        evaluator=None,
        train_split=None,
        eval_splits=None,
        do_threshold_search=True,
    ):
        args = self.args
        best_summary = {split: {"reward_mean": -1e9} for split in eval_splits}
        best_params = {}

        # Initialize OOD detector
        policy.initialize_ood_detector(args, envs["train"])

        feature_type = policy.feature_type
        batch_transform = None
        if isinstance(rollout_obs, Dataset):
            if get_global_variable("benchmark") != "procgen":
                raise ValueError(
                    "Streaming SVDD rollout training is only supported for Procgen."
                )
            if policy.clf_name != "DeepSVDD":
                raise ValueError(
                    "Streaming rollout training is only supported for DeepSVDD."
                )
            batch_transform = self._make_streaming_batch_transform(
                policy, envs, feature_type
            )
        else:
            rollout_obs = self._stack_rollout_obs(
                rollout_obs, "rollout_obs", feature_type
            )
        if val_rollout_obs is not None:
            val_rollout_obs = self._stack_rollout_obs(
                val_rollout_obs, "val_rollout_obs", feature_type
            )

        # Train OOD detector
        # x_threshold is currently being used to determine 'training' scores.
        # This could eventually be used to determine scores on an in-distribution
        # validation set that is different from the training set. Not sure if we
        # want to do this.
        policy.fit(
            x=rollout_obs,
            x_threshold=rollout_obs,
            x_val=val_rollout_obs,
            batch_transform=batch_transform,
        )

        # clf: AutoEncoderWithVal = policy.clf
        # val_scores = clf.val_score_list
        # training_scores = clf.training_score_list
        # if val_scores is not None:
        #     np.save(f"{self.save_dir}/val_scores.npy", np.array(val_scores))
        # if training_scores is not None:
        #     np.save(f"{self.save_dir}/training_scores.npy", np.array(training_scores))

        if do_threshold_search:
            # Threshold search
            thresholds_min, thresholds_max = (
                policy.clf.decision_scores_.min(),
                policy.clf.decision_scores_.max(),
            )
            if thresholds_min == thresholds_max:
                cand_thresholds = [thresholds_min]
            else:
                cand_thresholds = np.linspace(
                    thresholds_min, thresholds_max, args.num_thresholds
                )
            for threshold in cand_thresholds:
                params = {"threshold": threshold}
                logging.info(f"Evaluating threshold: {threshold}")

                policy.update_params(params)
                split_summary = evaluator.eval(policy, envs, eval_splits)

                for split in eval_splits:
                    if (
                        split_summary[split]["reward_mean"]
                        > best_summary[split]["reward_mean"]
                    ):
                        best_params[split] = params
                        best_summary[split] = split_summary[split]
                        policy.save_model(f"best_{split}", self.save_dir)

                    # Log best result so far
                    logging.info(f"Best {split} so far")
                    logging.info(f"Parameters: {best_params[split]}")
                    evaluator.write_summary(f"best_{split}", best_summary[split])

            policy.update_params(
                best_params[eval_splits[0]]
            )  # Update with best params from first eval split
        else:
            logging.info("Skipping threshold search.")

            logging.info("Saving trained model.")
            policy.save_model("trained", self.save_dir)

    def _stack_rollout_obs(self, rollout_obs, name, feature_type):
        # OODPolicy expects a single tensor, while rollout loading keeps samples as
        # a list to reduce peak memory before training starts.
        if not isinstance(rollout_obs, list):
            raise ValueError(
                f"Expected {name} to be a list of tensors, got "
                f"{type(rollout_obs)}, something might be wrong."
            )

        if get_global_variable("benchmark") == "minigrid" and feature_type not in [
            "hidden",
            "dist",
            "hidden_dist",
        ]:
            rollout_obs = rollout_obs[1::3]
            return torch.cat(rollout_obs, dim=0)
        return torch.stack(rollout_obs)

    def _make_streaming_batch_transform(self, policy, envs, feature_type):
        if feature_type == "obs":
            return None
        if feature_type != "hidden":
            raise ValueError(
                "Streaming DeepSVDD training currently supports feature_type='obs' "
                f"and feature_type='hidden', got {feature_type!r}."
            )

        weak_agent = envs["train"].weak_agent
        weak_agent.eval()
        model = getattr(weak_agent, "model", None)
        if model is not None:
            model.eval()
        agent_device = self._agent_device(weak_agent, policy.device)

        def hidden_batch_transform(batch):
            obs_batch = self._extract_single_tensor_batch(batch).to(agent_device)
            with torch.no_grad():
                return weak_agent.get_hidden(obs_batch).detach().cpu()

        return hidden_batch_transform

    def _extract_single_tensor_batch(self, batch):
        if torch.is_tensor(batch):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            return batch[0]
        raise ValueError(
            "Expected a tensor batch from the streaming rollout dataset, got "
            f"{type(batch)}."
        )

    def _agent_device(self, agent, default_device):
        if hasattr(agent, "_device"):
            return agent._device
        model = getattr(agent, "model", None)
        if model is not None:
            device = getattr(model, "device", None)
            if device is not None:
                return device
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass
        return default_device

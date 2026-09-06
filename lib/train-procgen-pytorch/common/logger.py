import numpy as np
import pandas as pd
from collections import deque
import time
import csv

try:
    import wandb
except ImportError:
    pass


def _mean_or_nan(values):
    return np.mean(values) if len(values) > 0 else np.nan


class Logger(object):
    def __init__(
        self,
        n_envs,
        logdir,
        use_wandb=False,
        log_interval=1000000,
        use_random_start_validation=False,
    ):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir
        self.use_wandb = use_wandb
        self.use_random_start_validation = use_random_start_validation

        # training
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])

        self.episode_timeout_buffer = deque(maxlen=40)
        self.episode_len_buffer = deque(maxlen=40)
        self.episode_reward_buffer = deque(maxlen=40)

        # validation
        self.episode_rewards_v = []
        for _ in range(n_envs):
            self.episode_rewards_v.append([])

        self.episode_timeout_buffer_v = deque(maxlen=40)
        self.episode_len_buffer_v = deque(maxlen=40)
        self.episode_reward_buffer_v = deque(maxlen=40)

        # random-start validation
        self.episode_rewards_v_random_start = []
        for _ in range(n_envs):
            self.episode_rewards_v_random_start.append([])

        self.episode_timeout_buffer_v_random_start = deque(maxlen=40)
        self.episode_len_buffer_v_random_start = deque(maxlen=40)
        self.episode_reward_buffer_v_random_start = deque(maxlen=40)

        time_metrics = ["timesteps", "wall_time", "num_episodes"]  # only collected once
        episode_metrics = [
            "max_episode_rewards",
            "mean_episode_rewards",
            "min_episode_rewards",
            "max_episode_len",
            "mean_episode_len",
            "min_episode_len",
            "mean_timeouts",
        ]  # collected for both train and val envs
        columns = time_metrics + episode_metrics + ["val_" + m for m in episode_metrics]
        if self.use_random_start_validation:
            columns += ["val_random_start_" + m for m in episode_metrics]
        columns += [
            "val_id_mean_episode_rewards",
            "val_ood_mean_episode_rewards",
            "val_id_num_episodes",
            "val_ood_num_episodes",
            "val_mean_oracle_regret",
            "val_id_mean_oracle_regret",
            "val_ood_mean_oracle_regret",
        ]
        self.log = pd.DataFrame(columns=columns)

        self.val_id_returns = []
        self.val_ood_returns = []
        self.val_oracle_regret = []
        self.val_id_oracle_regret = []
        self.val_ood_oracle_regret = []

        self.timesteps = 0
        self.num_episodes = 0
        self.log_interval = log_interval
        self.next_log_timestep = self.log_interval

    def feed(
        self,
        rew_batch,
        done_batch,
        rew_batch_v=None,
        done_batch_v=None,
        rew_batch_v_random_start=None,
        done_batch_v_random_start=None,
    ):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        valid_random_start = (
            self.use_random_start_validation
            and rew_batch_v_random_start is not None
            and done_batch_v_random_start is not None
        )
        if valid_random_start:
            rew_batch_v_random_start = rew_batch_v_random_start.T
            done_batch_v_random_start = done_batch_v_random_start.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])
                if valid_random_start:
                    self.episode_rewards_v_random_start[i].append(
                        rew_batch_v_random_start[i][j]
                    )

                if done_batch[i][j]:
                    self.episode_timeout_buffer.append(1 if j == steps - 1 else 0)
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    # Save returns
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    # Reset step rewards
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
                if valid and done_batch_v[i][j]:
                    self.episode_timeout_buffer_v.append(1 if j == steps - 1 else 0)
                    self.episode_len_buffer_v.append(len(self.episode_rewards_v[i]))
                    self.episode_reward_buffer_v.append(
                        np.sum(self.episode_rewards_v[i])
                    )
                    self.episode_rewards_v[i] = []
                if valid_random_start and done_batch_v_random_start[i][j]:
                    self.episode_timeout_buffer_v_random_start.append(
                        1 if j == steps - 1 else 0
                    )
                    self.episode_len_buffer_v_random_start.append(
                        len(self.episode_rewards_v_random_start[i])
                    )
                    self.episode_reward_buffer_v_random_start.append(
                        np.sum(self.episode_rewards_v_random_start[i])
                    )
                    self.episode_rewards_v_random_start[i] = []

        self.timesteps += self.n_envs * steps

    def feed_validation(
        self,
        episode_returns,
        episode_lengths,
        episode_timeouts=None,
        random_start=False,
        episode_ood=None,
        episode_regret=None,
    ):
        if episode_timeouts is None:
            episode_timeouts = [0] * len(episode_returns)

        if not (len(episode_returns) == len(episode_lengths) == len(episode_timeouts)):
            raise ValueError(
                "Validation returns, lengths, and timeouts must have matching lengths."
            )

        if random_start:
            reward_buffer = self.episode_reward_buffer_v_random_start
            len_buffer = self.episode_len_buffer_v_random_start
            timeout_buffer = self.episode_timeout_buffer_v_random_start
        else:
            reward_buffer = self.episode_reward_buffer_v
            len_buffer = self.episode_len_buffer_v
            timeout_buffer = self.episode_timeout_buffer_v

        for episode_return, episode_length, episode_timeout in zip(
            episode_returns, episode_lengths, episode_timeouts
        ):
            reward_buffer.append(float(episode_return))
            len_buffer.append(int(episode_length))
            timeout_buffer.append(int(episode_timeout))

        if random_start:
            return

        if episode_ood is not None:
            if len(episode_ood) != len(episode_returns):
                raise ValueError(
                    "Validation OOD flags must have the same length as episode returns."
                )

            for episode_return, is_ood in zip(episode_returns, episode_ood):
                if is_ood:
                    self.val_ood_returns.append(float(episode_return))
                else:
                    self.val_id_returns.append(float(episode_return))

        if episode_regret is None:
            return

        if len(episode_regret) != len(episode_returns):
            raise ValueError(
                "Validation regret values must have the same length as episode returns."
            )

        for idx, regret in enumerate(episode_regret):
            if regret is None:
                continue
            regret_value = float(regret)
            self.val_oracle_regret.append(regret_value)
            if episode_ood is None:
                continue
            if episode_ood[idx]:
                self.val_ood_oracle_regret.append(regret_value)
            else:
                self.val_id_oracle_regret.append(regret_value)

    def dump(self):
        if self.timesteps < self.next_log_timestep:
            return

        # Update the next log timestep.
        self.next_log_timestep += self.log_interval

        # Actually log the data.
        wall_time = time.time() - self.start_time
        episode_statistics = self._get_episode_statistics()
        episode_statistics_list = list(episode_statistics.values())
        log = [self.timesteps, wall_time, self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        with open(self.logdir + "/log-append.csv", "a") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(self.log.columns)
            writer.writerow(log)

        print(self.log.loc[len(self.log) - 1])

        if self.use_wandb:
            wandb.log({k: v for k, v in zip(self.log.columns, log)})

        # Reset the episode buffers.
        self.episode_timeout_buffer.clear()
        self.episode_len_buffer.clear()
        self.episode_reward_buffer.clear()

        self.episode_timeout_buffer_v.clear()
        self.episode_len_buffer_v.clear()
        self.episode_reward_buffer_v.clear()

        self.episode_timeout_buffer_v_random_start.clear()
        self.episode_len_buffer_v_random_start.clear()
        self.episode_reward_buffer_v_random_start.clear()
        self.val_id_returns = []
        self.val_ood_returns = []
        self.val_oracle_regret = []
        self.val_id_oracle_regret = []
        self.val_ood_oracle_regret = []

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics["Rewards/max_episodes"] = np.max(
            self.episode_reward_buffer, initial=0
        )
        episode_statistics["Rewards/mean_episodes"] = _mean_or_nan(
            self.episode_reward_buffer
        )
        episode_statistics["Rewards/min_episodes"] = np.min(
            self.episode_reward_buffer, initial=0
        )
        episode_statistics["Len/max_episodes"] = np.max(
            self.episode_len_buffer, initial=0
        )
        episode_statistics["Len/mean_episodes"] = _mean_or_nan(self.episode_len_buffer)
        episode_statistics["Len/min_episodes"] = np.min(
            self.episode_len_buffer, initial=0
        )
        episode_statistics["Len/mean_timeout"] = _mean_or_nan(
            self.episode_timeout_buffer
        )

        # valid
        episode_statistics["[Valid] Rewards/max_episodes"] = np.max(
            self.episode_reward_buffer_v, initial=0
        )
        episode_statistics["[Valid] Rewards/mean_episodes"] = _mean_or_nan(
            self.episode_reward_buffer_v
        )
        episode_statistics["[Valid] Rewards/min_episodes"] = np.min(
            self.episode_reward_buffer_v, initial=0
        )
        episode_statistics["[Valid] Len/max_episodes"] = np.max(
            self.episode_len_buffer_v, initial=0
        )
        episode_statistics["[Valid] Len/mean_episodes"] = _mean_or_nan(
            self.episode_len_buffer_v
        )
        episode_statistics["[Valid] Len/min_episodes"] = np.min(
            self.episode_len_buffer_v, initial=0
        )
        episode_statistics["[Valid] Len/mean_timeout"] = _mean_or_nan(
            self.episode_timeout_buffer_v
        )
        if self.use_random_start_validation:
            episode_statistics["[Valid Random Start] Rewards/max_episodes"] = np.max(
                self.episode_reward_buffer_v_random_start, initial=0
            )
            episode_statistics["[Valid Random Start] Rewards/mean_episodes"] = (
                _mean_or_nan(self.episode_reward_buffer_v_random_start)
            )
            episode_statistics["[Valid Random Start] Rewards/min_episodes"] = np.min(
                self.episode_reward_buffer_v_random_start, initial=0
            )
            episode_statistics["[Valid Random Start] Len/max_episodes"] = np.max(
                self.episode_len_buffer_v_random_start, initial=0
            )
            episode_statistics["[Valid Random Start] Len/mean_episodes"] = _mean_or_nan(
                self.episode_len_buffer_v_random_start
            )
            episode_statistics["[Valid Random Start] Len/min_episodes"] = np.min(
                self.episode_len_buffer_v_random_start, initial=0
            )
            episode_statistics["[Valid Random Start] Len/mean_timeout"] = _mean_or_nan(
                self.episode_timeout_buffer_v_random_start
            )
        episode_statistics["[Valid] Rewards/id_mean_episodes"] = _mean_or_nan(
            self.val_id_returns
        )
        episode_statistics["[Valid] Rewards/ood_mean_episodes"] = _mean_or_nan(
            self.val_ood_returns
        )
        episode_statistics["[Valid] Rewards/id_num_episodes"] = len(self.val_id_returns)
        episode_statistics["[Valid] Rewards/ood_num_episodes"] = len(
            self.val_ood_returns
        )
        episode_statistics["[Valid] Rewards/mean_oracle_regret"] = _mean_or_nan(
            self.val_oracle_regret
        )
        episode_statistics["[Valid] Rewards/id_mean_oracle_regret"] = _mean_or_nan(
            self.val_id_oracle_regret
        )
        episode_statistics["[Valid] Rewards/ood_mean_oracle_regret"] = _mean_or_nan(
            self.val_ood_oracle_regret
        )
        return episode_statistics

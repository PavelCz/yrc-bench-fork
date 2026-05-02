from .base_agent import BaseAgent
from common.misc_util import adjust_lr
import torch
import torch.optim as optim
import numpy as np


class PPO(BaseAgent):
    def __init__(
        self,
        env,
        policy,
        logger,
        storage,
        device,
        n_checkpoints,
        env_valid=None,
        storage_valid=None,
        n_steps=128,
        n_envs=8,
        epoch=3,
        mini_batch_per_epoch=8,
        mini_batch_size=32 * 8,
        gamma=0.99,
        lmbda=0.95,
        learning_rate=2.5e-4,
        grad_clip_norm=0.5,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        normalize_adv=True,
        normalize_rew=True,
        use_gae=True,
        log_interval=1000000,
        num_validation_episodes=1024,
        create_env_valid_fn=None,
        env_valid_random_start=None,
        storage_valid_random_start=None,
        create_env_valid_random_start_fn=None,
        **kwargs,
    ):
        super(PPO, self).__init__(
            env,
            policy,
            logger,
            storage,
            device,
            n_checkpoints,
            env_valid,
            storage_valid,
        )
        self.create_env_valid_fn = create_env_valid_fn
        self.env_valid_random_start = env_valid_random_start
        self.storage_valid_random_start = storage_valid_random_start
        self.create_env_valid_random_start_fn = create_env_valid_random_start_fn

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate, eps=1e-5
        )
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.log_interval = log_interval
        self.next_log_timestep = self.log_interval
        self.num_validation_episodes = num_validation_episodes
        self.validation_episode_returns = np.zeros(self.n_envs, dtype=np.float64)
        self.validation_episode_lengths = np.zeros(self.n_envs, dtype=np.int64)
        self.validation_random_start_episode_returns = np.zeros(
            self.n_envs, dtype=np.float64
        )
        self.validation_random_start_episode_lengths = np.zeros(
            self.n_envs, dtype=np.int64
        )

    def _reset_validation_accumulators(self, random_start=False):
        if random_start:
            self.validation_random_start_episode_returns.fill(0.0)
            self.validation_random_start_episode_lengths.fill(0)
        else:
            self.validation_episode_returns.fill(0.0)
            self.validation_episode_lengths.fill(0)

    def _get_validation_accumulators(self, random_start=False):
        if random_start:
            return (
                self.validation_random_start_episode_returns,
                self.validation_random_start_episode_lengths,
            )
        return self.validation_episode_returns, self.validation_episode_lengths

    def _raw_rewards(self, rewards, infos):
        if len(infos) > 0 and "env_reward" in infos[0]:
            return np.array([info["env_reward"] for info in infos], dtype=np.float64)
        return np.asarray(rewards, dtype=np.float64)

    def _episode_timeout(self, info):
        return bool(info.get("timeout", info.get("TimeLimit.truncated", False)))

    def run_validation(self, env, obs, hidden_state, done, random_start=False):
        num_episodes = 0
        episode_returns = []
        episode_lengths = []
        episode_timeouts = []
        current_returns, current_lengths = self._get_validation_accumulators(
            random_start=random_start
        )
        while num_episodes < self.num_validation_episodes:
            act, log_prob_act, value, next_hidden_state = self.predict(
                obs, hidden_state, done
            )
            next_obs, rew, done, info = env.step(act)
            del log_prob_act, value

            current_returns += self._raw_rewards(rew, info)
            current_lengths += 1
            done_indices = np.flatnonzero(done)
            for env_idx in done_indices:
                episode_returns.append(float(current_returns[env_idx]))
                episode_lengths.append(int(current_lengths[env_idx]))
                episode_timeouts.append(int(self._episode_timeout(info[env_idx])))
                current_returns[env_idx] = 0.0
                current_lengths[env_idx] = 0
            num_episodes += len(done_indices)

            obs = next_obs
            hidden_state = next_hidden_state

        return obs, hidden_state, done, {
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "episode_timeouts": episode_timeouts,
        }

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return (
            act.cpu().numpy(),
            log_prob_act.cpu().numpy(),
            value.cpu().numpy(),
            hidden_state.cpu().numpy(),
        )

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1 - done).to(device=self.device)
        dist, value, hidden_state = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return (
            act.detach().cpu().numpy(),
            log_prob_act.detach().cpu().numpy(),
            value.detach().cpu().numpy(),
            hidden_state.detach().cpu().numpy(),
            obs.grad.data.detach().cpu().numpy(),
        )

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(
                mini_batch_size=self.mini_batch_size, recurrent=recurrent
            )
            for sample in generator:
                (
                    obs_batch,
                    hidden_state_batch,
                    act_batch,
                    done_batch,
                    old_log_prob_act_batch,
                    old_value_batch,
                    return_batch,
                    adv_batch,
                ) = sample
                mask_batch = 1 - done_batch
                dist_batch, value_batch, _ = self.policy(
                    obs_batch, hidden_state_batch, mask_batch
                )

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * adv_batch
                )
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (
                    value_batch - old_value_batch
                ).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = (
                    pi_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.grad_clip_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {
            "Loss/pi": np.mean(pi_loss_list),
            "Loss/v": np.mean(value_loss_list),
            "Loss/entropy": np.mean(entropy_loss_list),
        }
        return summary

    def train(self, num_timesteps):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)
        if self.env_valid_random_start is not None:
            obs_v_random_start = self.env_valid_random_start.reset()
            hidden_state_v_random_start = np.zeros(
                (self.n_envs, self.storage.hidden_state_size)
            )
            done_v_random_start = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(
                    obs, hidden_state, done
                )
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(
                    obs, hidden_state, act, rew, done, info, log_prob_act, value
                )
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(
                self.gamma, self.lmbda, self.use_gae, self.normalize_adv
            )

            # valid
            next_timestep = self.t + self.n_steps * self.n_envs
            should_validate = next_timestep >= self.next_log_timestep
            if should_validate:
                self.next_log_timestep += self.log_interval

            if self.env_valid is not None and should_validate:
                # Re-create eval env if factory provided
                if self.create_env_valid_fn is not None:
                    self.env_valid.close()
                    self.env_valid = self.create_env_valid_fn()
                    obs_v = self.env_valid.reset()
                    hidden_state_v = np.zeros(
                        (self.n_envs, self.storage.hidden_state_size)
                    )
                    done_v = np.zeros(self.n_envs)
                    self._reset_validation_accumulators(random_start=False)

                # Run validation episodes.
                obs_v, hidden_state_v, done_v, validation_stats = self.run_validation(
                    self.env_valid,
                    obs_v,
                    hidden_state_v,
                    done_v,
                    random_start=False,
                )
                self.logger.feed_validation(
                    validation_stats["episode_returns"],
                    validation_stats["episode_lengths"],
                    validation_stats["episode_timeouts"],
                    random_start=False,
                )

            if self.env_valid_random_start is not None and should_validate:
                if self.create_env_valid_random_start_fn is not None:
                    self.env_valid_random_start.close()
                    self.env_valid_random_start = (
                        self.create_env_valid_random_start_fn()
                    )
                    obs_v_random_start = self.env_valid_random_start.reset()
                    hidden_state_v_random_start = np.zeros(
                        (self.n_envs, self.storage.hidden_state_size)
                    )
                    done_v_random_start = np.zeros(self.n_envs)
                    self._reset_validation_accumulators(random_start=True)

                (
                    obs_v_random_start,
                    hidden_state_v_random_start,
                    done_v_random_start,
                    validation_random_start_stats,
                ) = self.run_validation(
                    self.env_valid_random_start,
                    obs_v_random_start,
                    hidden_state_v_random_start,
                    done_v_random_start,
                    random_start=True,
                )
                self.logger.feed_validation(
                    validation_random_start_stats["episode_returns"],
                    validation_random_start_stats["episode_lengths"],
                    validation_random_start_stats["episode_timeouts"],
                    random_start=True,
                )

            # Optimize policy & valueq
            self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed(
                rew_batch,
                done_batch,
            )
            self.logger.dump()
            self.optimizer = adjust_lr(
                self.optimizer, self.learning_rate, self.t, num_timesteps
            )
            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every):
                print("Saving model.")
                torch.save(
                    {
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.logger.logdir + "/model_" + str(self.t) + ".pth",
                )
                checkpoint_cnt += 1
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()
        if self.env_valid_random_start is not None:
            self.env_valid_random_start.close()

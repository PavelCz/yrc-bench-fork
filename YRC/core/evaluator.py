import logging
from matplotlib.path import Path
import numpy as np
import os
from PIL import Image


class Evaluator:
    LOGGED_ACTION = 1

    def __init__(self, config, env_name="", seed=None, observation_dir=None):
        self.args = config
        self.env_name = env_name
        self.iter = 0
        self.run_id= seed
        self.observation_dir = observation_dir

    def eval(self, policy, envs, eval_splits, num_episodes=None):
        args = self.args
        policy.eval()

        # Create directories for saving obvservations
        if self.env_name != "" and self.observation_dir is not None:
            output_dir = Path(self.observation_dir)
            img_dir = output_dir / self.env_name
            img_dir.mkdir(parents=True, exist_ok=True)
        else:
            img_dir = None
        
        summary = {}
        for split in eval_splits:
            if num_episodes is None:
                if "val" in split:
                    num_episodes = args.validation_episodes
                else:
                    assert "test" in split
                    num_episodes = args.test_episodes
                assert num_episodes % envs[split].num_envs == 0

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            num_iterations = num_episodes // envs[split].num_envs
            log = {}
            for _ in range(num_iterations):
                this_log = self._eval_one_iteration(policy, envs[split], img_dir)
                self._update_log(log, this_log)

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            envs[split].close()

        return summary

    def _update_log(self, log, this_log):
        if not log:
            log.update(this_log)
        for k, v in this_log.items():
            if isinstance(v, list):
                log[k].extend(v)
            else:
                log[k] += v
    def _eval_one_iteration(self, policy, env, img_dir: Path | None):
        args = self.args
        log = {
            "reward": [0] * env.num_envs,
            "env_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.LOGGED_ACTION}": 0,
        }

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        step = 0

        all_obs = [[] for _ in range(env.num_envs)]
        step_counts = [0 for _ in range(env.num_envs)]

        while not has_done.all():
            # Save images, if it's a test run. The hacky way I'm doing that is if env_name is nonempty
            if self.env_name != "":
                for env_idx in range(env.num_envs):
                    if not has_done[env_idx]:
                        # Get observation for this environment
                        obs_array = obs['env_obs'][env_idx] if isinstance(obs, dict) else obs[env_idx]

                        # Transpose from (C, H, W) to (H, W, C)
                        obs_array = np.transpose(obs_array, (1, 2, 0))

                        # Convert to uint8 for PIL
                        obs_array = (obs_array * 255).astype(np.uint8)

                        img = Image.fromarray(obs_array)
                        img.save(os.path.join(img_dir, f'iter{self.iter}_env{env_idx}_step{step_counts[env_idx]}_run-id{self.run_id}.png'))
                        step_counts[env_idx] += 1

            action = policy.act(obs, greedy=args.act_greedy)
            obs, reward, done, info= env.step(action)

            for i in range(env.num_envs):
                if "env_reward" in info[i]:
                    log["env_reward"][i] += info[i]["env_reward"] * (1 - has_done[i])

                log["reward"][i] += reward[i] * (1 - has_done[i])
                log["episode_length"][i] += 1 - has_done[i]
                if not has_done[i]:
                    log[f"action_{self.LOGGED_ACTION}"] += (action[i] == self.LOGGED_ACTION).sum()

            has_done |= done
            step += 1

        self.iter += 1
        return log

    def summarize(self, log):
        return {
            "steps": int(sum(log["episode_length"])),
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "reward_mean": float(np.mean(log["reward"])),
            "raw_reward": log["reward"],
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            f"action_{self.LOGGED_ACTION}_frac": float(
                log[f"action_{self.LOGGED_ACTION}"] / sum(log["episode_length"])
            ),
        }

    def write_summary(self, split, summary):
        log_str = f"   Steps:       {summary['steps']}\n"
        log_str += "   Episode:    "
        log_str += f"mean {summary['episode_length_mean']:7.2f}  "
        log_str += f"min {summary['episode_length_min']:7.2f}  "
        log_str += f"max {summary['episode_length_max']:7.2f}\n"
        log_str += "   Reward:     "
        log_str += f"mean {summary['reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['env_reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += f"   Action {self.LOGGED_ACTION} fraction: {summary[f'action_{self.LOGGED_ACTION}_frac']:7.2f}\n"
        log_str += "   Raw Rewards: "
        for r in summary["raw_reward"]:
            log_str += f"{r:.2f},"
        logging.info(log_str)

        return summary

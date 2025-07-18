import logging
import numpy as np
from typing import Optional
from pytorch_lightning.loggers import WandbLogger
import wandb

class Evaluator:
    LOGGED_ACTION = 1

    def __init__(self, config):
        self.args = config
        self.collected_states = []
        self.collected_actions_done = False

    def eval(
        self, 
        policy, 
        envs, 
        eval_splits, 
        num_episodes=None, 
        logger: Optional[WandbLogger] = None, 
        threshold: Optional[float] = None,
        percentile_step: Optional[float] = None,
    ):
        args = self.args
        policy.eval()

        self.collected_actions_done = False
        self.collected_states = []

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

            log = self._eval_loop(policy, envs[split], num_episodes)

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            envs[split].close()

        if logger is not None:

            obs = [x["obs"] for x in self.collected_states]
            scores = [x["scores"] for x in self.collected_states]
            recons = [x["recons"] for x in self.collected_states]
            action = [x["action"] for x in self.collected_states]

            vid = np.stack(obs, axis=1)
            vid = vid * 255
            vid = vid.astype(np.int8)
            logger.experiment.log(
                {
                    f"eval_episode_{threshold:.2f}": wandb.Video(
                        # (batch dim, time dim, c, h, w)
                        vid,
                        fps=15,
                        format="gif",
                        caption=(
                            f"Threshold: {threshold:.2f}, "
                            # f"Percentile: {percentile_step:.2f}"
                        ),
                    ),
                }
            )

        return summary

    def _eval_loop(self, policy, env, max_episodes: int) -> dict:
        args = self.args

        log = {
            "reward": [],
            "env_reward": [],
            "episode_length": [],
            f"action_{self.LOGGED_ACTION}": [],
        }

        # A temporary log that only contains stats for the current episode.
        episode_log = {
            "reward": [0] * env.num_envs,
            "env_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.LOGGED_ACTION}": [0] * env.num_envs,
        }

        obs = env.reset()

        # This tracks the very first done and is only used to determine whether to keep
        # collecting observations that are later used to generate the video.
        has_done = np.array([False] * env.num_envs)
        num_episodes = 0

        while num_episodes < max_episodes:

            # For most policies I have seen, the greedy flag is ignored. These include
            # random, lightning_ae, and ood.
            action, scores, recons = policy.act(
                obs, greedy=args.act_greedy, return_scores_and_recons=True
            )

            if not all(has_done):
                self.collected_states.append({
                    "obs": obs["env_obs"],
                    "scores": scores,
                    "recons": recons,
                    "action": action,
                })

            obs, reward, done, info = env.step(action)

            for i in range(env.num_envs):

                if "env_reward" in info[i]:
                    episode_log["env_reward"][i] += info[i]["env_reward"]

                episode_log["reward"][i] += reward[i]
                episode_log["episode_length"][i] += 1
                episode_log[f"action_{self.LOGGED_ACTION}"][i] += (action[i] == self.LOGGED_ACTION).sum()

                if done[i] and num_episodes < max_episodes:
                    log["reward"].append(episode_log["reward"][i])
                    log["env_reward"].append(episode_log["env_reward"][i])
                    log["episode_length"].append(episode_log["episode_length"][i])
                    log[f"action_{self.LOGGED_ACTION}"].append(episode_log[f"action_{self.LOGGED_ACTION}"][i])
                    num_episodes += 1

                    episode_log["reward"][i] = 0
                    episode_log["env_reward"][i] = 0
                    episode_log["episode_length"][i] = 0
                    episode_log[f"action_{self.LOGGED_ACTION}"][i] = 0

            has_done |= done

        return log

    def summarize(self, log):
        total_steps = int(sum(log["episode_length"]))
        return {
            "steps": total_steps,
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "reward_mean": float(np.mean(log["reward"])),
            "raw_reward": log["reward"],
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            f"action_{self.LOGGED_ACTION}_frac": float(
                sum(log[f"action_{self.LOGGED_ACTION}"]) / total_steps
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

import jsonargparse


def make():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, help="path to YAML config file")
    parser.add_argument("-d", "--general.device", type=int, help="device id")
    parser.add_argument(
        "-wandb",
        "--use_wandb",
        action="store_true",
        default=False,
        help="log to wandb?",
    )

    # wandb settings
    parser.add_argument(
        "-wandb_project",
        "--wandb.project",
        type=str,
        default="yrc-01",
        help="wandb project name",
    )
    parser.add_argument(
        "-wandb_group",
        "--wandb.group",
        type=str,
        help="wandb group name (defaults to exp_name if not specified)",
    )
    parser.add_argument(
        "-wandb_mode",
        "--wandb.mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="wandb mode: online, offline, or disabled",
    )
    parser.add_argument(
        "-wandb_entity", "--wandb.entity", type=str, help="wandb entity/username"
    )

    parser.add_argument(
        "-no_eval",
        "--algorithm.no_eval",
        action="store_true",
        default=False,
        help="no evaluation",
    )
    parser.add_argument(
        "-log_freq", "--algorithm.log_freq", type=int, help="Frequency of logging"
    )
    parser.add_argument(
        "-clip_vloss", "--algorithm.clip_vloss", type=int, help="Clip value loss (RL)"
    )
    parser.add_argument(
        "-norm_adv", "--algorithm.norm_adv", type=int, help="Normalize advantage (RL)"
    )
    parser.add_argument("-n", "--name", type=str, help="name of this run")
    parser.add_argument(
        "-over",
        "--overwrite",
        action="store_true",
        help="overwrite experiment folder (if exists)",
    )
    parser.add_argument(
        "-query_cost",
        "--coord_env.strong_query_cost_ratio",
        type=float,
        help="Cost of querying strong agent",
    )
    parser.add_argument(
        "-switch_cost",
        "--coord_env.switch_agent_cost_ratio",
        type=float,
        help="Cost of switching agent",
    )
    parser.add_argument(
        "-en", "--environment.common.env_name", type=str, help="name of the environment"
    )
    parser.add_argument(
        "-sim", "--agents.sim_weak", type=str, help="path to the sim weak agent"
    )
    parser.add_argument(
        "-weak", "--agents.weak", type=str, help="path to the weak agent"
    )
    parser.add_argument(
        "-strong", "--agents.strong", type=str, help="path to the strong agent"
    )
    parser.add_argument(
        "-f_n", "--file_name", type=str, help="file name for evaluation"
    )
    parser.add_argument(
        "-agent",
        "--general.agent",
        type=str,
        choices=["weak", "strong"],
        help="agent to evaluate",
    )
    parser.add_argument(
        "-model_file",
        "--model_file",
        type=str,
        help="path to model file to evaluate (for eval_policy.py)",
    )
    parser.add_argument(
        "-greedy",
        "--policy.greedy",
        type=bool,
        default=True,
        help=(
            "use greedy action selection during evaluation (default: True)"
            "Only used in eval_policy.py"
        ),
    )
    parser.add_argument(
        "-cp_feature",
        "--coord_policy.feature_type",
        type=str,
        choices=[
            "obs",
            "hidden",
            "hidden_obs",
            "dist",
            "hidden_dist",
            "obs_dist",
            "obs_hidden_dist",
        ],
        help="Type of features for coordination policy",
    )
    parser.add_argument(
        "-cp_data_agent",
        "--coord_policy.collect_data_agent",
        type=str,
        choices=["weak", "strong"],
        default="weak",
        help="agent to collect data",
    )

    # always policy
    parser.add_argument(
        "-cp_agent",
        "--coord_policy.agent",
        type=str,
        choices=["weak", "strong"],
        help="always choose action of this agent",
    )

    # threshold policy
    parser.add_argument(
        "-cp_metric",
        "--coord_policy.metric",
        type=str,
        choices=["max_logit", "max_prob", "margin", "neg_entropy", "neg_energy"],
        help="metric for computing scores",
    )

    # ood policy
    parser.add_argument(
        "-cp_method",
        "--coord_policy.method",
        type=str,
        choices=["DeepSVDD", "AutoEncoder", "Autoencoder"],
        help="method for detecting OOD samples",
    )

    parser.add_argument(
        "-cp_latent_dim",
        "--coord_policy.latent_dim",
        type=int,
        help="latent dimension for the autoencoder",
    )

    parser.add_argument(
        "-cp_rolling_average",
        "--coord_policy.rolling_average",
        type=str,
        help="rolling average for the threshold policy",
        choices=["mean", "median", "none"],
    )
    parser.add_argument(
        "-cp_rolling_average_size",
        "--coord_policy.rolling_average_size",
        type=int,
        help="size of the rolling average for the threshold policy",
    )

    parser.add_argument(
        "-model_config_path",
        "--algorithm.model_config_path",
        type=str,
        help="path to the model config file",
    )

    parser.add_argument(
        "-disable_test",
        "--algorithm.disable_test",
        action="store_true",
        help="disable PyTorch Lightning test run.",
    )

    parser.add_argument(
        "-cp_epoch",
        "--algorithm.epoch",
        type=int,
        help=(
            "Number of epochs for training the OOD detector. "
            "This overrides the epoch in the config file."
        ),
    )

    # random baseline policy
    parser.add_argument(
        "-cp_base",
        "--coord_policy.baseline",
        action="store_true",
        help="baseline policy with random action 0.5 probability",
    )

    # minigrid
    parser.add_argument(
        "-en_tr_suffix",
        "--environment.train.env_name_suffix",
        type=str,
        help="suffix for the train environment name",
    )
    parser.add_argument(
        "-en_te_suffix",
        "--environment.test.env_name_suffix",
        type=str,
        help="suffix for the test environment name",
    )

    # procgen
    parser.add_argument(
        "-use_bg",
        "--environment.common.use_backgrounds",
        type=bool,
        default=True,
        help="use background - only for procgen envs",
    )
    parser.add_argument(
        "-use_mono_asset",
        "--environment.common.use_monochrome_assets",
        type=bool,
        default=False,
        help="use monochrome assets - only for procgen envs",
    )
    parser.add_argument(
        "-res_theme",
        "--environment.common.restrict_themes",
        type=bool,
        default=False,
        help="restrict themes - only for procgen envs",
    )
    parser.add_argument(
        "-max_steps",
        "--environment.common.max_steps",
        type=int,
        help="maximum number of timesteps per episode (max: 1000) - only for procgen envs",
    )

    parser.add_argument("-seed", "--general.seed", type=int, help="random seed")

    parser.add_argument(
        "-num_rollouts",
        "--algorithm.num_rollouts",
        type=int,
        help="number of rollouts to collect for training",
    )

    parser.add_argument(
        "-batch_size",
        "--algorithm.batch_size",
        type=int,
        help="batch size for training the OOD detector",
    )

    parser.add_argument(
        "-num_envs",
        "--environment.common.num_envs",
        type=int,
        help="number of environments to run in parallel",
    )

    parser.add_argument(
        "-random_percent",
        "--environment.test.random_percent",
        type=int,
        help="random percent for the test environment",
    )

    # Additional flags for evaluation.
    parser.add_argument(
        "-coverage_fraction",
        "--evaluation.coverage_fraction",
        type=float,
        help=(
            "The maximum gap allowed between consecutive points, as a fraction of the "
            "total range. Choose at least 0.01"
        ),
    )
    parser.add_argument(
        "-threshold_sampler",
        "--evaluation.threshold_sampler",
        type=str,
        choices=["afhp", "ood_percentage"],
        help="threshold sampler to use",
    )

    # parser.add_argument(
    #     "-test_episodes",
    #     "--evaluation.test_episodes",
    #     type=int,
    #     help="number of test episodes for procgen",
    # )
    parser.add_argument(
        "-num_levels",
        "--environment.test.num_levels",
        type=int,
        help=(
            "number of test levels for procgen. The same number is used to determine "
            "the number of episodes for evaluation."
        ),
    )

    parser.add_argument(
        "-defer_to_oracle",
        "--evaluation.defer_to_oracle",
        action="store_true",
        help="defer to oracle for evaluation",
    )

    parser.add_argument(
        "-eval_run_name",
        "--eval_run_name",
        type=str,
        help=(
            "custom name for the evaluation run (timestamp will be appended "
            "automatically)"
        ),
    )

    parser.add_argument(
        "-video_logging_mode",
        "--evaluation.video_logging_mode",
        type=str,
        choices=["wandb", "folder", "both", "none"],
        default="folder",
        help="video logging mode: wandb (Weights & Biases), folder (local files), both, or none",
    )

    parser.add_argument(
        "-video_output_folder",
        "--evaluation.video_output_folder",
        type=str,
        help="folder path for saving videos when using folder or both logging modes (relative to eval_run_dir, defaults to eval_run_dir/videos if not specified)",
    )

    parser.add_argument(
        "-experiment_group",
        "--experiment_group",
        type=str,
        help=(
            "experiment group name - used for wandb group and as prefix for eval names"
        ),
    )

    parser.add_argument(
        "-video_episodes_to_collect",
        "--evaluation.video_episodes_to_collect",
        type=int,
        help="Number of video episodes to collect for evaluation",
    )

    parser.add_argument(
        "-video_filter",
        "--evaluation.video_filter",
        type=str,
        nargs="+",
        help="Filter criteria for which episodes to save as videos. Can specify multiple filters to create separate categories. Options: 'all', 'no_death', 'random_coin_success', 'deterministic_coin_success', 'ood_detected', 'in_distribution'",
    )

    parser.add_argument(
        "-video_filter_mode",
        "--evaluation.video_filter_mode",
        type=str,
        choices=["any", "all"],
        default="any",
        help="Filter mode: 'any' creates separate category for each filter, 'all' requires episode to pass all filters to be saved",
    )

    parser.add_argument(
        "-rollout_dir",
        "--training.rollout_dir",
        type=str,
        help="directory to save rollouts",
    )

    parser.add_argument(
        "-val_rollout_dir",
        "--training.val_rollout_dir",
        type=str,
        help="directory to save val rollouts. Currently only used for Mahalanobis AE.",
    )

    args = parser.parse_args()

    return args

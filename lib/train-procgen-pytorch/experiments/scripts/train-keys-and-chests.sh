#!/bin/bash
set -o errexit

# this script trains on the heist_aisc_many_chests environment
# applying a small penalty for picking up keys.
#
# NOTE (yrc-bench-fork): key_penalty=3 and param_name=A100 are script knobs.
# They are not the paper's keys-and-chests mainline description, and they differ
# from our coinrun/maze/heist baseline in scripts/train_policies.sh
# (param_name=paper, key_penalty=0). See GitHub issue #23.

experiment_name="key-penalty"
key_penalty=3

let n_steps=80*10**6
n_checkpoints=4
n_threads=32 # 32 CPUs per GPU
wandb_tags="hpc large-model"

keys_and_chests_opt="--env_name heist_aisc_many_chests --val_env_name heist_aisc_many_keys --key_penalty $key_penalty"


# include the option
#       --model_file auto
# when resuming a run from a saved checkpoint. This should load the latest model
# saved under $experiment_name

options="
	$keys_and_chests_opt
	--param_name A100
	--use_wandb
	--distribution_mode hard
	--num_timesteps $n_steps
	--num_checkpoints $n_checkpoints
	--num_threads $n_threads
	--wandb_tags $wandb_tags
	--exp_name $experiment_name
        "

python train.py $options

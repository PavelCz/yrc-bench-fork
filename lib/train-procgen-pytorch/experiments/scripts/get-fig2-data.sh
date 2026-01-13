#!/bin/bash
set -o errexit

# gather metrics for figure 2
# load trained coinrun models and deploy them in the test environment.
# to do this without specifying the model_file every time, trained coinrun
# models must be stored in logs with exp_name 'freq-sweep-random-percent-$random_percent'
# write output metrics to csv files in ./experiments/results/

if [ $# -lt 3 ]; then
    echo "Usage: $0 <standard|joint> <model_file> <random_percent> [num_seeds]"
    echo "  model_file: path to model file or an integer (interpreted as training random_percent)"
    echo "  num_seeds defaults to 10000"
    exit 1
fi

mode="$1"
model_file="$2"
random_percent="$3"
num_seeds="${4:-10000}"

if [[ $mode = 'standard' ]]
then
    python run_coinrun.py --model_file "$model_file" --start_level_seed 0 --num_seeds "$num_seeds" --random_percent 100
elif [[ $mode = 'joint' ]]
then
    python run_coinrun.py --model_file "$model_file" --start_level_seed 0 --num_seeds "$num_seeds" --random_percent "$random_percent" --reset_mode "complete"
fi

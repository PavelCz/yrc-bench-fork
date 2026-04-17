# Explanation of the files in this directory

## `yrc_bench`

This directory contains the analysis code for the YRC benchmark.
Kept here for reference.

## `afhp_plot.py`

Plot the AFHP -> performance curve.

## `analyze_environment_structure.py`

I use this to do some debugging to better understand the 
procgen environments.

## `coinrun_counterfactual_analysis.py`

Finds levels where the weak agent fails and rolls out the weak agent in two versions
of the environment: one with deterministic coin placement and one with random coin placement.

## `plot_policy_training_curves.py`

Plot the policy training timestep -> reward curve.

## `policy_eval_plot.py`

Aggregate `eval_policy.py` JSON outputs across experiment IDs, print overall / ID /
OOD return summaries, and optionally save a grouped bar plot.

## `plotting.ipynb`

Plotting notebook.

## `utils.py`

Utility functions for these analyses.

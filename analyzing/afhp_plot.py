from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_path = Path("/home/pavel/data/goal-misgen/tmp")


eval_path = base_path / "26-5000" 

evals = {} 

for child in eval_path.iterdir():
    if child.is_dir():
        method_name = child.name
        if (child / "eval_runs").exists():
            for grandchild in (child / "eval_runs").iterdir():
                for grandgrandchild in grandchild.iterdir():
                    if grandgrandchild.is_file() and grandgrandchild.suffix == ".npz":
                        evals[method_name] = grandgrandchild

# Collect first and last performance values for all curves
first_performances = []
last_performances = []

name_map = {
    "latent-svdd": "Latent SVDD",
    "random": "Random",
    "patient-ae": "Autoencoder",
    "center-focused": "Center-focused AE",
    "deep-svdd": "DeepSVDD",
}

all = False
if all:
    name_order = list(evals.keys())
else:
    name_order = ["random", "deep-svdd", "patient-ae", "center-focused", "latent-svdd"]

for name in name_order:
    data_path = evals[name]

    eval_data = np.load(data_path, allow_pickle=True)

    afhps = eval_data["afhps"]
    performances = eval_data["performances"]
    desired_percentiles = eval_data["desired_percentiles"]
    
    # Store first and last performance values
    first_performances.append(performances[0])
    last_performances.append(performances[-1])

    if name in name_map:
        label = name_map[name]
    else:
        label = name
    
    sns.lineplot(x=afhps, y=performances, label=label, marker="o")

# Calculate means
mean_first_performance = np.mean(first_performances)
mean_last_performance = np.mean(last_performances)

# Add horizontal lines
plt.axhline(y=mean_first_performance, color='red', linestyle='--', alpha=0.7, 
            label=f'Weak Agent')
plt.axhline(y=mean_last_performance, color='blue', linestyle='--', alpha=0.7, 
            label=f'Oracle')

plt.xlabel("Ask for help percentage")
plt.ylabel("Mean return")
plt.title("Mean return vs. ask for help percentage")
plt.legend()
plt.show()

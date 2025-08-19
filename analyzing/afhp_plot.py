from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_path = Path("/home/pavel/data/goal-misgen/tmp")


eval_path = base_path / "26-5000" 

evals = []
names = []

for child in eval_path.iterdir():
    if child.is_dir():
        method_name = child.name
        if (child / "eval_runs").exists():
            for grandchild in (child / "eval_runs").iterdir():
                for grandgrandchild in grandchild.iterdir():
                    if grandgrandchild.is_file() and grandgrandchild.suffix == ".npz":
                        evals.append(grandgrandchild)
                        names.append(method_name)

# Collect first and last performance values for all curves
first_performances = []
last_performances = []

for name, eval in zip(names, evals):

    data_path = eval

    eval_data = np.load(data_path, allow_pickle=True)

    afhps = eval_data["afhps"]
    performances = eval_data["performances"]
    desired_percentiles = eval_data["desired_percentiles"]
    
    # Store first and last performance values
    first_performances.append(performances[0])
    last_performances.append(performances[-1])
    
    sns.lineplot(x=afhps, y=performances, label=name, marker="o")

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

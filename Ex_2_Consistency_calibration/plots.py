import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gmean
import os
from pathlib import Path


# -------------------------------------------------------
# --------------- PLOT N.1: ECE histogram ---------------
# -------------------------------------------------------

fig, axis = plt.subplots(1, 1, figsize=(15, 8))  # create subplots

result_path = Path(__file__).resolve().parent / "Results"
plot_path = Path(__file__).resolve().parent / "Plots"
project_folder =  Path(__file__).resolve().parent.parent

# Read naive and consistency df
naive_df = pd.read_csv(result_path / "chronos-t5-small_naive.csv")
consistency_df = pd.read_csv(
    result_path / "chronos-t5-small_std_2_00_npert_128_consistency.csv"
)

# Get ECE from eache df
naive_ece = naive_df["ECE"]
consistency_ece = consistency_df["ECE"]

print(naive_ece.std())
print(consistency_ece.std())

# Compute upper_edge
upper_edge = max(max(naive_ece), max(consistency_ece))

# Create bins from 0 to upper_edge with step 0.02
bins = np.arange(0, upper_edge + 0.02, 0.02)  # includes the upper edge


# Plot The histograms
# fmt:off
axis.hist(consistency_ece, bins=bins, edgecolor="black", alpha=0.9, label="Calibrated (std 2)",  zorder=3)
axis.hist(naive_ece, bins=bins, edgecolor="black", alpha=0.7, label="Naive", zorder=3)
# fmt: on


# Set axis labels, title, grid and legend
axis.set_xlabel("ECE", fontsize=30)
axis.set_ylabel("Number of datasets", fontsize=30)
# axis.set_title("ECE Distribution", fontsize=40)
axis.grid(zorder=0, alpha=0.5)
axis.legend(fontsize=25)
axis.tick_params(axis="both", labelsize=12)

# Save and show the plot
plt.show()
os.makedirs(str(plot_path), exist_ok=True)
fig.savefig(plot_path / "histogram.png", bbox_inches="tight")


# ---------------------------------------------------------
# ----------------------- Lineplots -----------------------
# ---------------------------------------------------------


std_list = [2, 4, 8]
pert_list = [16, 32, 64, 128]
base_filename = "chronos-t5-small_std_{:.2f}_npert_{}_consistency"
baseline_file = project_folder / "scripts" / "evaluation" / "results"/ "seasonal-naive-zero-shot.csv"
baseline_df = pd.read_csv(baseline_file)
titles = [
    "MASE",
    "WQL",
    "ECE",
]

# Remove df that was not included in testing, drop textual columns and reset index
baseline_df = baseline_df[baseline_df["dataset"] != "m5"]
baseline_df = baseline_df.drop(["model", "dataset"], axis="columns")
baseline_df = baseline_df.reset_index(drop=True)

relative_score = (
    naive_df.drop(["dataset", "model", "ECE"], axis="columns") / baseline_df
)
original_mase = [gmean(relative_score["MASE"])] * len(std_list)
original_wql = [gmean(relative_score["WQL"])] * len(std_list)
original_ece = [gmean(naive_df["ECE"])] * len(std_list)

for i in range(3):
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    stat_name = titles[i].split()[0]
    # axis.set_title(titles[i], fontsize=50)
    # axis.set_ylabel(titles[i].split()[0], fontsize=30)
    axis.tick_params(axis="both", labelsize=20)
    axis.set_xlabel(
        r"Noise strength (values of $\sigma$ in the Gaussian noise)", fontsize=30
    )
    axis.set_xticks(range(len(std_list)), labels=std_list)
    axis.grid(True, linewidth=0.3, color="gray")

    for pert in pert_list:
        stat = []
        naive_stat = None
        for std in std_list:

            full_name = base_filename.format(std, pert).replace(".", "_") + ".csv"
            df = pd.read_csv(result_path / full_name)
            if titles[i].startswith("ECE"):
                stat.append(gmean(df["ECE"]))
                naive_stat = original_ece
            else:
                relative_score = (
                    df.drop(["dataset", "model", "ECE"], axis="columns") / baseline_df
                )
                if titles[i].startswith("MASE"):
                    stat.append(gmean(relative_score["MASE"]))
                    naive_stat = original_mase
                else:
                    stat.append(gmean(relative_score["WQL"]))
                    naive_stat = original_wql

            # print("--------------------")
            # print("STD: ", std)
            # print("NPERT:", pert))
            # stat.append(gmean(relative_score["MASE"]))
            # stat.append(gmean(relative_score["WQL"]))
        axis.plot(
            stat, label=f"Perturbations: {pert}", marker="s", linewidth=5, markersize=8
        )
    axis.plot(naive_stat, label="Original", linestyle="dashed", linewidth=5)

    axis.legend(fontsize=25)

    name = titles[i][:4].strip()
    fig.savefig(plot_path / f"{name}.png", bbox_inches="tight")
plt.show()

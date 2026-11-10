import os
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "./results/tables"
FIGURES_DIR = "./results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

experiment_configs = [
    {
        "label": "FedAvg Baseline (Stationary)",
        "file": "fedavg_mnist_baseline_metrics.csv",
        "color": "#1f77b4",       # Blue
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.4,
        "alpha": 0.9,
    },
    {
        "label": "Personalized Fine-Tune (Stationary)",
        "file": "personalized_finetune_mnist_metrics.csv",
        "color": "#ff7f0e",       # Orange
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.4,
        "alpha": 0.9,
    },
    {
        "label": "Fixed Policy (Abrupt Drift)",
        "file": "exp_abrupt_fixed_fixed_ddm_s42_metrics.csv",
        "color": "#d62728",       # Red
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.4,
        "alpha": 0.9,
    },
    {
        "label": "Drift-Aware Policy (Abrupt Drift)",
        "file": "exp_abrupt_drift_aware_fixed_ddm_s42_metrics.csv",
        "color": "#2ca02c",       # Green
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.6,
        "alpha": 0.9,
    },
    {
        "label": "Drift-Triggered Comm (Abrupt Drift)",
        "file": "exp_abrupt_drift_aware_drift_triggered_ddm_s42_metrics.csv",
        "color": "#9467bd",       # Purple
        "marker": "v",
        "linestyle": ":",
        "linewidth": 2.6,
        "alpha": 0.9,
    },
]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

for cfg in experiment_configs:
    path = os.path.join(RESULTS_DIR, cfg["file"])
    if os.path.exists(path):
        df = pd.read_csv(path)
        rounds = df["round"]
        acc_col = "mean_client_test_accuracy" if "mean_client_test_accuracy" in df.columns else "global_accuracy"
        accs = df[acc_col] * 100
        comm_mb = df["cumulative_comm_bytes"] / (1024 * 1024)

        ax1.plot(
            rounds,
            accs,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            markersize=7,
            alpha=cfg["alpha"],
        )
        ax2.plot(
            rounds,
            comm_mb,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            markersize=7,
            alpha=cfg["alpha"],
        )

# Left plot: Accuracy
ax1.set_title("(a) Client Test Accuracy Across Rounds under Abrupt Drift", fontsize=12.5, fontweight="bold", pad=10)
ax1.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
ax1.set_ylabel("Mean Client Test Accuracy (%)", fontsize=11, fontweight="bold")
ax1.set_ylim(-3, 105)
ax1.set_xticks(range(1, 6))
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.2)

# Add drift onset marker annotation
ax1.axvline(x=2.0, color="#8b0000", linestyle=":", linewidth=1.6)
ax1.text(2.06, 18, "Abrupt Drift Onset\n(Round 2)", color="#8b0000", fontsize=9.5, fontweight="bold")

# Right plot: Cumulative Bandwidth
ax2.set_title("(b) Cumulative Network Bandwidth Consumed", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
ax2.set_ylabel("Cumulative Transfer (MB)", fontsize=11, fontweight="bold")
ax2.set_xticks(range(1, 6))
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.2)

plt.tight_layout()

fig_path1 = os.path.join(FIGURES_DIR, "drift_comparative_evaluation.png")
fig_path2 = os.path.join("./actual_paper", "drift_comparative_evaluation.png")
plt.savefig(fig_path1, dpi=300)
plt.savefig(fig_path2, dpi=300)
print(f"Updated high-visibility figure saved to {fig_path1} and {fig_path2}")

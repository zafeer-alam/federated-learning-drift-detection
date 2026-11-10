import os
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "./results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

# Subplot (a): Post-Drift Accuracy across all 4 drift regimes (Fixed vs Drift-Aware + Pool)
regimes = ["Abrupt", "Gradual", "Recurring", "Staggered"]
fixed_accs = [44.58, 51.24, 62.15, 58.40]
drift_aware_accs = [96.81, 94.12, 97.09, 95.34]

x = np.arange(len(regimes))
width = 0.35

rects1 = ax1.bar(x - width / 2, fixed_accs, width, label="Fixed Selection Policy", color="#d95f02", alpha=0.9)
rects2 = ax1.bar(x + width / 2, drift_aware_accs, width, label="Drift-Aware + Retention Pool", color="#1b9e77", alpha=0.9)

ax1.set_ylabel("Mean Client Test Accuracy (%)", fontsize=11, fontweight="bold")
ax1.set_title("(a) Post-Drift Accuracy Across Drift Regimes", fontsize=12.5, fontweight="bold", pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(regimes, fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 122)
ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
ax1.grid(axis="y", linestyle="--", alpha=0.6)

for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Subplot (b): Cumulative 5-Round Bandwidth Footprint Across Scheduling Policies (K=3 clients)
policies = ["Fixed Periodic\n(tau=1)", "Hybrid Schedule\n(tau=2 + Drift)", "Drift-Triggered\n(Alarm Only)"]
bandwidth_mb = [190.35, 152.28, 126.90]
colors = ["#4C72B0", "#DD8452", "#55A868"]

bars = ax2.bar(policies, bandwidth_mb, color=colors, width=0.48, alpha=0.9)
ax2.set_ylabel("5-Round Cumulative Bandwidth (MB)", fontsize=11, fontweight="bold")
ax2.set_title("(b) Communication Overhead Across Scheduling Policies", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_ylim(0, 220)
ax2.grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 4,
        f"{yval:.2f} MB",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

plt.tight_layout()
out1 = os.path.join(FIGURES_DIR, "ablation_analysis.png")
out2 = os.path.join("./actual_paper", "ablation_analysis.png")
plt.savefig(out1, dpi=300)
plt.savefig(out2, dpi=300)
print(f"Saved ablation figure to {out1} and {out2}")

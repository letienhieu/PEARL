"""Generate training-curve figure for paper Section 5 (results) or method.

Reads booksim_build/exports/training_history_credit_aware.csv and produces
fig_training_curve.png showing loss + replay-buffer growth + episode-mean
latency over the 500-episode training run.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "booksim_build/exports/training_history_credit_aware.csv"
OUT = ROOT / "results_dddqn/fig_training_curve.png"

chunks, lats, eps_vals, bufs, losses = [], [], [], [], []
with open(HIST) as f:
    for row in csv.DictReader(f):
        chunks.append(int(row["chunk_end"]))
        lats.append(float(row["avg_lat"]))
        eps_vals.append(float(row["eps"]))
        bufs.append(int(row["buf"]))
        losses.append(float(row["loss"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax1 = axes[0]
ax1.plot(chunks, losses, color="#d62728", lw=1.2, label="Bellman loss")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Mean Bellman loss (per 8-episode chunk)", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.grid(True, alpha=0.3)
ax1b = ax1.twinx()
ax1b.plot(chunks, eps_vals, color="#1f77b4", lw=1.0, linestyle="--", label="$\\epsilon$")
ax1b.set_ylabel("$\\epsilon$ (exploration)", color="#1f77b4")
ax1b.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_title("(a) Loss + $\\epsilon$ schedule")

ax2 = axes[1]
ax2.plot(chunks, lats, color="#2ca02c", lw=0.8, alpha=0.4, label="Per-chunk")
window = 5  # running-average window ≈ 40 episodes
if len(lats) >= window:
    smoothed = np.convolve(lats, np.ones(window)/window, mode="valid")
    smooth_x = chunks[window-1:]
    ax2.plot(smooth_x, smoothed, color="#1a6020", lw=1.8,
             label=f"Running avg (w={window} chunks)")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Episode-mean BookSim packet latency (cycles)")
ax2.set_title("(b) Episode-mean latency")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"✓ {OUT}")

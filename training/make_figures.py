"""Generate paper figures from results_dddqn/aggregated_metrics.csv.

Outputs (PNG, 300 DPI for publication):
  results_dddqn/fig_latency_curves.png — 3×3 grid of latency-vs-IR per (mesh, traffic)
  results_dddqn/fig_rank_heatmap.png   — 6 methods × 45 cells (3×3 mesh-traffic grid)
  results_dddqn/fig_win_rate.png       — bar chart of "best method" wins per method
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_dddqn"

# Load aggregated CSV
agg: dict[tuple, dict] = {}
with open(RESULTS / "aggregated_metrics.csv") as f:
    for row in csv.DictReader(f):
        key = (row["method"], int(row["k"]), row["traffic"], float(row["ir"]))
        agg[key] = {"mean": float(row["mean"]), "std": float(row["std"]),
                    "n": int(row["n"]), "ci95": float(row["ci95"])}

METHODS = ["XY", "OddEven", "DyAD", "MinAdapt", "QRouting", "DDDQN", "CreditMinAdapt"]
COLORS = {"XY": "#888888", "OddEven": "#1f77b4", "DyAD": "#2ca02c",
          "MinAdapt": "#ff7f0e", "QRouting": "#d62728", "DDDQN": "#9467bd",
          "CreditMinAdapt": "#e377c2"}
MARKERS = {"XY": "o", "OddEven": "s", "DyAD": "^", "MinAdapt": "v",
           "QRouting": "D", "DDDQN": "*", "CreditMinAdapt": "P"}
TARGET_METHOD = "CreditMinAdapt"  # method to highlight in rank heatmap
MESHES = [4, 8, 16]
TRAFFIC = ["uniform", "hotspot", "transpose"]
IRS = sorted({k_[3] for k_ in agg.keys()})

# ===========================================================================
# Figure 1: Latency vs IR, 3×3 grid
# ===========================================================================
fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
for i, k in enumerate(MESHES):
    for j, t in enumerate(TRAFFIC):
        ax = axes[i, j]
        for m in METHODS:
            xs, ys, errs = [], [], []
            for ir in IRS:
                key = (m, k, t, ir)
                if key in agg:
                    xs.append(ir)
                    ys.append(agg[key]["mean"])
                    errs.append(agg[key]["ci95"])
            ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[m], color=COLORS[m],
                        label=m, lw=1.3, ms=6,
                        markeredgecolor="black" if m == TARGET_METHOD else None,
                        markeredgewidth=0.5 if m == TARGET_METHOD else 0)
        ax.set_title(f"{k}×{k}  /  {t}", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Injection rate (packets/node/cycle)")
        if j == 0:
            ax.set_ylabel("Avg packet latency (cycles)")
axes[0, 2].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
plt.tight_layout()
fig_path = RESULTS / "fig_latency_curves.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ {fig_path}")

# ===========================================================================
# Figure 2: Rank heatmap — DDDQN's rank in each cell
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))
cells = []
for k in MESHES:
    for t in TRAFFIC:
        row = []
        for ir in IRS:
            lats = [(m, agg[(m, k, t, ir)]["mean"]) for m in METHODS
                    if (m, k, t, ir) in agg]
            lats.sort(key=lambda x: x[1])
            try:
                rank_d = next(i + 1 for i, (m, _) in enumerate(lats) if m == TARGET_METHOD)
            except StopIteration:
                rank_d = np.nan
            row.append(rank_d)
        cells.append(row)
M = np.asarray(cells)
im = ax.imshow(M, cmap="RdYlGn_r", vmin=1, vmax=7, aspect="auto")
ax.set_xticks(range(len(IRS)))
ax.set_xticklabels([f"{ir:.3f}" for ir in IRS])
ax.set_yticks(range(9))
ax.set_yticklabels([f"{k}×{k} {t}" for k in MESHES for t in TRAFFIC])
ax.set_xlabel("Injection rate")
ax.set_ylabel("(mesh, traffic)")
ax.set_title(f"{TARGET_METHOD} Rank in {len(METHODS)}-Method Comparison (1=best)")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] >= 5 else "black", fontsize=10)
plt.colorbar(im, ax=ax, label="Rank")
plt.tight_layout()
fig_path = RESULTS / "fig_rank_heatmap.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ {fig_path}")

# ===========================================================================
# Figure 3: Win-rate bar chart
# ===========================================================================
# Win-counting with ε-threshold (Δ > 0.5 cycle).
# When 2+ methods are within 0.5 cycle of best, they share the win.
# This prevents ε-greedy noise + identical policy behavior from artificially
# inflating the win count of either tied method.
EPSILON_TIE = 0.5  # cycles
win_count = defaultdict(float)
for k in MESHES:
    for t in TRAFFIC:
        for ir in IRS:
            cell_lats = [(m, agg[(m, k, t, ir)]["mean"]) for m in METHODS
                         if (m, k, t, ir) in agg]
            if not cell_lats:
                continue
            best_l = min(l for _, l in cell_lats)
            tied_winners = [m for m, l in cell_lats if l <= best_l + EPSILON_TIE]
            share = 1.0 / len(tied_winners)
            for m in tied_winners:
                win_count[m] += share

fig, ax = plt.subplots(figsize=(8, 5))
methods_sorted = sorted(METHODS, key=lambda m: -win_count[m])
counts = [win_count[m] for m in methods_sorted]
bars = ax.bar(methods_sorted, counts,
              color=[COLORS[m] for m in methods_sorted],
              edgecolor="black", linewidth=0.5)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, c + 0.3, f"{c:.1f}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Best-in-cell wins (ties shared, n=45 total)")
ax.set_title(f"Best-in-cell wins across 45 cells (ε-tie={EPSILON_TIE} cycle)")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(counts) + 3)
plt.tight_layout()
fig_path = RESULTS / "fig_win_rate.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ {fig_path}")
print()
print("Win counts (ε-threshold tie-sharing):")
for m in methods_sorted:
    print(f"  {m:<16}: {win_count[m]:>5.1f} / 45")

"""mn-1: Throughput (accepted packet rate) vs injection rate curves.

Reads results_dddqn/aggregated_metrics.csv (need to add accepted_pkt_rate from
per_run CSV since aggregated only has avg_lat).
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_dddqn"

# Load per-run CSV to get throughput (accepted_pkt_rate)
runs = defaultdict(list)
with open(RESULTS / "per_run_metrics.csv") as f:
    for row in csv.DictReader(f):
        try:
            rate = float(row["accepted_pkt_rate"])
        except (ValueError, KeyError):
            continue
        if rate <= 0: continue
        key = (row["method"], int(row["k"]), row["traffic"], float(row["ir"]))
        runs[key].append(rate)

# Aggregate
agg_rate = {k: (np.mean(v), 1.96 * np.std(v) / np.sqrt(len(v))) for k, v in runs.items()}

METHODS = ["XY", "OddEven", "DyAD", "MinAdapt", "QRouting", "DDDQN", "CreditMinAdapt"]
COLORS = {"XY": "#888888", "OddEven": "#1f77b4", "DyAD": "#2ca02c",
          "MinAdapt": "#ff7f0e", "QRouting": "#d62728", "DDDQN": "#9467bd",
          "CreditMinAdapt": "#e377c2"}
MARKERS = {"XY": "o", "OddEven": "s", "DyAD": "^", "MinAdapt": "v",
           "QRouting": "D", "DDDQN": "*", "CreditMinAdapt": "P"}
TARGET = "CreditMinAdapt"
MESHES = [4, 8, 16]
TRAFFIC = ["uniform", "hotspot", "transpose"]
IRS = sorted({k[3] for k in agg_rate})

fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
for i, k in enumerate(MESHES):
    for j, t in enumerate(TRAFFIC):
        ax = axes[i, j]
        for m in METHODS:
            xs, ys, errs = [], [], []
            for ir in IRS:
                key = (m, k, t, ir)
                if key in agg_rate:
                    xs.append(ir)
                    ys.append(agg_rate[key][0])
                    errs.append(agg_rate[key][1])
            ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[m], color=COLORS[m],
                        label=m, lw=1.3, ms=6,
                        markeredgecolor="black" if m == TARGET else None,
                        markeredgewidth=0.5 if m == TARGET else 0)
        ax.set_title(f"{k}×{k}  /  {t}", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Injection rate (offered)")
        if j == 0:
            ax.set_ylabel("Accepted packet rate")
axes[0, 2].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
plt.tight_layout()
out = RESULTS / "fig_throughput_curves.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ {out}")

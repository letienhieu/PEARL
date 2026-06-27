"""MC-5: Action distribution heatmap per (mesh, traffic, IR) cell.

Runs DDDQN with logging enabled for each cell, parses the log, and computes
% of action 0 (XY) vs action 1 (adaptive) chosen. Visualizes as heatmap.
"""
from __future__ import annotations
import csv
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "booksim_build"
BOOKSIM = str(BUILD / "booksim")
RESULTS = ROOT / "results_dddqn"
WEIGHTS = str(BUILD / "exports/ddqn_credit_aware_4x4_500ep.txt")

MESHES = [4, 8, 16]
TRAFFIC = ["uniform", "hotspot", "transpose"]
IRS = [0.005, 0.01, 0.02, 0.04, 0.08]


def make_config(k, traffic, ir):
    return f"""topology=mesh; k={k}; n=2;
routing_function=dddqn;
num_vcs=4; vc_buf_size=8; wait_for_tail_credit=1;
vc_allocator=islip; sw_allocator=islip; alloc_iters=1;
credit_delay=2; routing_delay=0; vc_alloc_delay=1; sw_alloc_delay=1;
input_speedup=2; output_speedup=1;
traffic={traffic}; packet_size=20; sim_type=latency;
injection_rate={ir}; sample_period=2000; warmup_periods=2; max_samples=2;
seed=42;
"""


def collect_action_dist(k, traffic, ir):
    cfg = f"/tmp/actdist_{k}_{traffic}_{ir}.cfg"
    log = f"/tmp/actdist_{k}_{traffic}_{ir}.log"
    Path(cfg).write_text(make_config(k, traffic, ir))
    env = os.environ.copy()
    env["DDDQN_WEIGHTS"] = WEIGHTS
    env["DDDQN_LOG"] = log
    try:
        subprocess.run([BOOKSIM, cfg], cwd=str(BUILD), capture_output=True,
                       text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired:
        return 0.0, 0
    a0 = a1 = 0
    if not os.path.exists(log):
        return 0.0, 0
    with open(log) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if len(parts) < 27: continue
            try:
                act = int(parts[4])
                if act == 0: a0 += 1
                elif act == 1: a1 += 1
            except ValueError:
                continue
    n = a0 + a1
    return (100 * a1 / n if n else 0.0), n


# Collect 3x3 grid x 5 IR = 45 cells
print("Collecting action distribution per cell...")
data = {}
for k in MESHES:
    for t in TRAFFIC:
        for ir in IRS:
            pct_a1, n = collect_action_dist(k, t, ir)
            data[(k, t, ir)] = (pct_a1, n)
            print(f"  {k}x{k} {t} ir={ir}: action_1={pct_a1:.1f}% (n={n})")

# Build 9-row × 5-col matrix
cells = []
for k in MESHES:
    for t in TRAFFIC:
        row = [data[(k, t, ir)][0] for ir in IRS]
        cells.append(row)
M = np.asarray(cells)

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(IRS)))
ax.set_xticklabels([f"{ir:.3f}" for ir in IRS])
ax.set_yticks(range(9))
ax.set_yticklabels([f"{k}×{k} {t}" for k in MESHES for t in TRAFFIC])
ax.set_xlabel("Injection rate")
ax.set_ylabel("(mesh, traffic)")
ax.set_title("DDDQN Action 1 (Credit-Aware Adaptive) % per Cell\n(0% = pure XY, 100% = pure adaptive)")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                color="white" if (v < 25 or v > 75) else "black", fontsize=10)
plt.colorbar(im, ax=ax, label="% Action 1")
plt.tight_layout()
out = RESULTS / "fig_action_distribution.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ {out}")

# Save CSV for reference
csv_out = RESULTS / "action_distribution.csv"
with open(csv_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mesh", "traffic", "ir", "action_1_pct", "decisions_count"])
    for k in MESHES:
        for t in TRAFFIC:
            for ir in IRS:
                pct, n = data[(k, t, ir)]
                w.writerow([f"{k}x{k}", t, ir, round(pct, 2), n])
print(f"✓ {csv_out}")

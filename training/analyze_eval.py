"""Analyze eval matrix results: DDDQN vs all baselines, by mesh/traffic/IR."""
from __future__ import annotations
from pathlib import Path
import csv
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_dddqn"

# Load aggregated CSV
agg: dict[tuple, dict] = {}
with open(RESULTS / "aggregated_metrics.csv") as f:
    for row in csv.DictReader(f):
        key = (row["method"], int(row["k"]), row["traffic"], float(row["ir"]))
        agg[key] = {"mean": float(row["mean"]), "std": float(row["std"]),
                    "n": int(row["n"]), "ci95": float(row["ci95"])}

METHODS = ["XY", "OddEven", "DyAD", "MinAdapt", "QRouting", "DDDQN"]
MESHES = [4, 8, 16]
TRAFFIC = ["uniform", "hotspot", "transpose"]
IRS = [0.005, 0.01, 0.02, 0.04, 0.08]

# ========================================================================
# Table 1: DDDQN ranking among 6 methods
# ========================================================================
print("=" * 76)
print("Table 1: DDDQN's RANK among 6 methods (1=best, 6=worst)")
print("=" * 76)
print(f"{'k':>3} | {'traffic':<10} | " + " | ".join(f"{ir:>6.3f}" for ir in IRS) + " | rank_avg")
print("-" * 76)
rank_by_cell = []
for k in MESHES:
    for t in TRAFFIC:
        ranks = []
        cells = []
        for ir in IRS:
            lats = []
            for m in METHODS:
                key = (m, k, t, ir)
                if key in agg:
                    lats.append((m, agg[key]["mean"]))
            lats.sort(key=lambda x: x[1])  # ascending: best first
            rank_dddqn = next(i + 1 for i, (m, _) in enumerate(lats) if m == "DDDQN")
            ranks.append(rank_dddqn)
            cells.append(f"{rank_dddqn:>6}")
            rank_by_cell.append(rank_dddqn)
        avg_rank = np.mean(ranks)
        print(f"{k:>3} | {t:<10} | " + " | ".join(cells) + f" | {avg_rank:>5.2f}")

print(f"\nDDDQN average rank across all 45 cells: {np.mean(rank_by_cell):.2f} / 6.0")
print(f"DDDQN top-rank (=1) cells: {sum(1 for r in rank_by_cell if r == 1)}/45")
print(f"DDDQN top-2 cells: {sum(1 for r in rank_by_cell if r <= 2)}/45")
print(f"DDDQN bottom-rank (=6) cells: {sum(1 for r in rank_by_cell if r == 6)}/45")

# ========================================================================
# Table 2: Best method per cell
# ========================================================================
print()
print("=" * 76)
print("Table 2: BEST method per (k, traffic, ir) — count of DDDQN wins")
print("=" * 76)
best_counts = defaultdict(int)
print(f"{'k':>3} | {'traffic':<10} | " + " | ".join(f"{ir:>10.3f}" for ir in IRS))
print("-" * 90)
for k in MESHES:
    for t in TRAFFIC:
        cells = []
        for ir in IRS:
            best_m, best_l = None, float("inf")
            for m in METHODS:
                key = (m, k, t, ir)
                if key in agg and agg[key]["mean"] < best_l:
                    best_m, best_l = m, agg[key]["mean"]
            best_counts[best_m] += 1
            mark = "*" if best_m == "DDDQN" else " "
            cells.append(f"{mark}{best_m:>9}")
        print(f"{k:>3} | {t:<10} | " + " | ".join(cells))

print()
print("Win count by method (best in cell):")
for m in METHODS:
    print(f"  {m:<10}: {best_counts[m]:>2} / 45 cells")

# ========================================================================
# Table 3: DDDQN improvement over best classical baseline (best of XY/OddEven/DyAD/MinAdapt/QRouting)
# ========================================================================
print()
print("=" * 76)
print("Table 3: DDDQN vs BEST CLASSICAL BASELINE (% latency reduction)")
print("=" * 76)
print(f"{'k':>3} | {'traffic':<10} | " + " | ".join(f"{ir:>7.3f}" for ir in IRS))
print("-" * 76)
all_deltas = []
classical = [m for m in METHODS if m != "DDDQN"]
for k in MESHES:
    for t in TRAFFIC:
        cells = []
        for ir in IRS:
            d = agg.get(("DDDQN", k, t, ir))
            best_classical = min(
                (agg[("X", k, t, ir)]["mean"] for X in classical
                 if (X, k, t, ir) in agg),
                default=None,
            ) if False else None
            # Actually compute correctly
            best_classical_lat = None
            for m in classical:
                key = (m, k, t, ir)
                if key in agg:
                    if best_classical_lat is None or agg[key]["mean"] < best_classical_lat:
                        best_classical_lat = agg[key]["mean"]
            if d and best_classical_lat is not None:
                pct = 100 * (d["mean"] - best_classical_lat) / best_classical_lat
                all_deltas.append(pct)
                cells.append(f"{pct:>+6.2f}%")
            else:
                cells.append("    n/a")
        print(f"{k:>3} | {t:<10} | " + " | ".join(cells))

deltas = np.asarray(all_deltas)
print(f"\nDDDQN vs best-classical: mean Δ = {deltas.mean():+.2f}%  median = {np.median(deltas):+.2f}%")
print(f"DDDQN wins (Δ < 0): {(deltas < 0).sum()}/45  ({100*(deltas<0).mean():.1f}%)")
print(f"Largest DDDQN wins: {sorted(deltas)[:5]}")
print(f"Largest DDDQN losses: {sorted(deltas)[-5:]}")

# ========================================================================
# Save Table 3 to CSV for paper
# ========================================================================
out = RESULTS / "summary_dddqn_vs_best_classical.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k", "traffic", "ir", "dddqn_mean", "best_classical_method",
                "best_classical_mean", "delta_pct", "win"])
    for k in MESHES:
        for t in TRAFFIC:
            for ir in IRS:
                d = agg.get(("DDDQN", k, t, ir))
                best_m, best_l = None, None
                for m in classical:
                    key = (m, k, t, ir)
                    if key in agg:
                        if best_l is None or agg[key]["mean"] < best_l:
                            best_m, best_l = m, agg[key]["mean"]
                if d and best_l is not None:
                    pct = 100 * (d["mean"] - best_l) / best_l
                    w.writerow([k, t, ir, round(d["mean"], 4), best_m,
                                round(best_l, 4), round(pct, 2),
                                "yes" if pct < 0 else "no"])
print(f"\n✓ {out}")

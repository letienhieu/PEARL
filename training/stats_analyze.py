"""Phase 3: Statistical analysis with paired t-test and 95% CI.

Reads:
  results_dddqn/per_run_metrics.csv (1350 rows from 5-seed eval)

Outputs:
  results_dddqn/stats_dddqn_vs_minadapt.csv  — paired t-test per cell
  results_dddqn/stats_dddqn_vs_best.csv      — paired t-test vs best classical per cell
  results_dddqn/stats_summary.txt            — overall paper-ready summary
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as scstats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_dddqn"

# Load per-run CSV and group by (method, k, traffic, ir, seed)
runs = defaultdict(dict)  # (k, traffic, ir, seed) -> {method: avg_lat}
with open(RESULTS / "per_run_metrics.csv") as f:
    for row in csv.DictReader(f):
        try:
            avg_lat = float(row["avg_lat"]) if row["avg_lat"] != "inf" else float("inf")
        except ValueError:
            continue
        if not np.isfinite(avg_lat):
            continue
        key = (int(row["k"]), row["traffic"], float(row["ir"]), int(row["seed"]))
        runs[key][row["method"]] = avg_lat

METHODS = ["XY", "OddEven", "DyAD", "MinAdapt", "QRouting", "DDDQN"]
classical = [m for m in METHODS if m != "DDDQN"]
MESHES = [4, 8, 16]
TRAFFIC = ["uniform", "hotspot", "transpose"]
IRS = sorted({k_[2] for k_ in runs.keys()})


def paired_test(samples_a: list[float], samples_b: list[float]) -> tuple[float, float, float, float]:
    """Return (mean_a, mean_b, mean_diff, p_value) using paired t-test."""
    a = np.asarray(samples_a)
    b = np.asarray(samples_b)
    if len(a) < 2 or len(a) != len(b):
        return float(a.mean()) if len(a) else float("nan"), \
               float(b.mean()) if len(b) else float("nan"), \
               float("nan"), float("nan")
    t, p = scstats.ttest_rel(a, b)
    return float(a.mean()), float(b.mean()), float((a - b).mean()), float(p)


def ci95(samples: list[float]) -> tuple[float, float]:
    """Return (mean, ±half-width) for 95% CI."""
    arr = np.asarray(samples)
    if len(arr) < 2:
        return float(arr.mean()) if len(arr) else float("nan"), 0.0
    sem = scstats.sem(arr)
    return float(arr.mean()), float(sem * scstats.t.ppf(0.975, len(arr) - 1))


# ---------------------------------------------------------------------------
# Per-cell analysis — DDDQN vs MinAdapt + DDDQN vs best classical
# ---------------------------------------------------------------------------
out_minadapt = RESULTS / "stats_dddqn_vs_minadapt.csv"
out_best = RESULTS / "stats_dddqn_vs_best.csv"

with open(out_minadapt, "w", newline="") as fm, open(out_best, "w", newline="") as fb:
    wm = csv.writer(fm)
    wb = csv.writer(fb)
    wm.writerow(["k", "traffic", "ir", "n_seeds",
                 "dddqn_mean", "dddqn_ci95",
                 "minadapt_mean", "minadapt_ci95",
                 "delta_mean", "delta_pct", "p_value", "significant_p05",
                 "win"])
    wb.writerow(["k", "traffic", "ir", "n_seeds",
                 "dddqn_mean", "dddqn_ci95",
                 "best_classical_method", "best_mean", "best_ci95",
                 "delta_mean", "delta_pct", "p_value", "significant_p05",
                 "win"])

    n_signif_wins_minadapt = 0
    n_signif_losses_minadapt = 0
    n_total_minadapt = 0
    n_signif_wins_best = 0
    n_signif_losses_best = 0
    n_total_best = 0

    for k in MESHES:
        for t in TRAFFIC:
            for ir in IRS:
                # Collect per-seed paired samples for each method
                samples = {m: [] for m in METHODS}
                for seed in sorted({s_[3] for s_ in runs.keys()}):
                    key = (k, t, ir, seed)
                    if key not in runs:
                        continue
                    cell = runs[key]
                    if all(m in cell for m in METHODS):
                        for m in METHODS:
                            samples[m].append(cell[m])
                n = len(samples["DDDQN"])
                if n < 2:
                    continue

                d_mean, d_ci = ci95(samples["DDDQN"])
                # vs MinAdapt
                m_mean, m_ci = ci95(samples["MinAdapt"])
                _, _, diff_m, p_m = paired_test(samples["DDDQN"], samples["MinAdapt"])
                pct_m = 100 * diff_m / m_mean if m_mean else 0
                win_m = "yes" if d_mean < m_mean else "no"
                signif_m = "yes" if (np.isfinite(p_m) and p_m < 0.05) else "no"
                if signif_m == "yes":
                    if win_m == "yes":
                        n_signif_wins_minadapt += 1
                    else:
                        n_signif_losses_minadapt += 1
                n_total_minadapt += 1
                wm.writerow([k, t, f"{ir:.3f}", n,
                             round(d_mean, 4), round(d_ci, 4),
                             round(m_mean, 4), round(m_ci, 4),
                             round(diff_m, 4), round(pct_m, 2),
                             f"{p_m:.4g}" if np.isfinite(p_m) else "n/a",
                             signif_m, win_m])

                # vs best classical (per seed)
                # For each seed, find the classical-method min
                best_per_seed = []
                best_method_count = defaultdict(int)
                for i in range(n):
                    cell_lats = [(m, samples[m][i]) for m in classical]
                    best_m, best_l = min(cell_lats, key=lambda x: x[1])
                    best_per_seed.append(best_l)
                    best_method_count[best_m] += 1
                # Most common best method
                best_m_dom = max(best_method_count, key=best_method_count.get)
                b_mean, b_ci = ci95(best_per_seed)
                _, _, diff_b, p_b = paired_test(samples["DDDQN"], best_per_seed)
                pct_b = 100 * diff_b / b_mean if b_mean else 0
                win_b = "yes" if d_mean < b_mean else "no"
                signif_b = "yes" if (np.isfinite(p_b) and p_b < 0.05) else "no"
                if signif_b == "yes":
                    if win_b == "yes":
                        n_signif_wins_best += 1
                    else:
                        n_signif_losses_best += 1
                n_total_best += 1
                wb.writerow([k, t, f"{ir:.3f}", n,
                             round(d_mean, 4), round(d_ci, 4),
                             best_m_dom,
                             round(b_mean, 4), round(b_ci, 4),
                             round(diff_b, 4), round(pct_b, 2),
                             f"{p_b:.4g}" if np.isfinite(p_b) else "n/a",
                             signif_b, win_b])

print(f"✓ Per-cell stats vs MinAdapt → {out_minadapt}")
print(f"✓ Per-cell stats vs best classical → {out_best}")
print()
print("=" * 72)
print(f"DDDQN vs MinAdapt (n={n_total_minadapt} cells, paired t-test):")
print(f"  Significant wins  (Δ<0, p<0.05): {n_signif_wins_minadapt}/{n_total_minadapt}")
print(f"  Significant losses(Δ>0, p<0.05): {n_signif_losses_minadapt}/{n_total_minadapt}")
print(f"  Non-significant (tied)        : "
      f"{n_total_minadapt - n_signif_wins_minadapt - n_signif_losses_minadapt}/{n_total_minadapt}")
print()
print(f"DDDQN vs BEST classical (per-seed; n={n_total_best} cells):")
print(f"  Significant wins  (p<0.05): {n_signif_wins_best}/{n_total_best}")
print(f"  Significant losses(p<0.05): {n_signif_losses_best}/{n_total_best}")
print(f"  Non-significant (tied)    : "
      f"{n_total_best - n_signif_wins_best - n_signif_losses_best}/{n_total_best}")

# ---------------------------------------------------------------------------
# Save text summary
# ---------------------------------------------------------------------------
summary_path = RESULTS / "stats_summary.txt"
with open(summary_path, "w") as f:
    f.write("DDDQN Statistical Analysis Summary (5-seed paired t-test)\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Total cells: {n_total_minadapt} (mesh × traffic × IR)\n\n")
    f.write("vs MinAdapt:\n")
    f.write(f"  Significant wins  (p<0.05): {n_signif_wins_minadapt}/{n_total_minadapt}\n")
    f.write(f"  Significant losses(p<0.05): {n_signif_losses_minadapt}/{n_total_minadapt}\n\n")
    f.write("vs Best Classical (per-seed):\n")
    f.write(f"  Significant wins  (p<0.05): {n_signif_wins_best}/{n_total_best}\n")
    f.write(f"  Significant losses(p<0.05): {n_signif_losses_best}/{n_total_best}\n")
print(f"\n✓ Summary text → {summary_path}")

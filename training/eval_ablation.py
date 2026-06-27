"""Ablation eval: DDDQN (full Double DQN) vs no-Double DDDQN.

Per topic spec ablation scope: 8×8 mesh only, 3 traffic × 5 IRs × 5 seeds = 75 runs each.
Compares Double Q-learning's contribution while keeping all other components identical
(same Dueling architecture, same credit-aware action 1, same reward shaping v2).
"""
from __future__ import annotations
import csv
import multiprocessing as mp
import os
import re
import subprocess
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "booksim_build"
BOOKSIM = str(BUILD / "booksim")
RESULTS = ROOT / "results_dddqn"

DDDQN_WEIGHTS_FULL  = str(BUILD / "exports/ddqn_credit_aware_4x4_500ep.txt")  # Double
DDDQN_WEIGHTS_NODOUB = str(BUILD / "exports/ddqn_no_double_ablation_500ep.txt")  # vanilla target

VARIANTS = {"DDDQN_full": DDDQN_WEIGHTS_FULL, "DDDQN_no_double": DDDQN_WEIGHTS_NODOUB}
TRAFFIC = ["uniform", "hotspot", "transpose"]
IR_LIST = [0.005, 0.01, 0.02, 0.04, 0.08]
SEEDS = [42, 1337, 2024, 314, 17]


def make_config(k, traffic, ir, seed):
    return f"""topology=mesh; k={k}; n=2;
routing_function=dddqn;
num_vcs=4; vc_buf_size=8; wait_for_tail_credit=1;
vc_allocator=islip; sw_allocator=islip; alloc_iters=1;
credit_delay=2; routing_delay=0; vc_alloc_delay=1; sw_alloc_delay=1;
input_speedup=2; output_speedup=1;
traffic={traffic}; packet_size=20; sim_type=latency;
injection_rate={ir}; sample_period=2000; warmup_periods=2; max_samples=2;
seed={seed};
"""


def run_one(args):
    variant, k, traffic, ir, seed = args
    weights = VARIANTS[variant]
    cfg = make_config(k, traffic, ir, seed)
    cfg_path = f"/tmp/abl_{variant}_{k}_{traffic}_{ir}_{seed}.cfg"
    Path(cfg_path).write_text(cfg)
    env = os.environ.copy()
    env["DDDQN_WEIGHTS"] = weights
    try:
        res = subprocess.run([BOOKSIM, cfg_path], cwd=str(BUILD), capture_output=True,
                             text=True, timeout=180, env=env)
        out = res.stdout
    except subprocess.TimeoutExpired:
        return dict(variant=variant, k=k, traffic=traffic, ir=ir, seed=seed,
                    avg_lat=float("inf"), status="timeout")
    m = re.findall(r"Packet latency average\s*=\s*([\d.]+)", out)
    avg_lat = float(m[-1]) if m else float("inf")
    return dict(variant=variant, k=k, traffic=traffic, ir=ir, seed=seed,
                avg_lat=avg_lat, status="ok" if np.isfinite(avg_lat) else "no_output")


def main():
    configs = [(v, 8, t, ir, s)
               for v in VARIANTS
               for t in TRAFFIC
               for ir in IR_LIST
               for s in SEEDS]
    print(f"Total ablation runs: {len(configs)} (8×8 mesh only)")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(8) as pool:
        results = list(pool.imap_unordered(run_one, configs))
    elapsed_min = (time.time() - t0) / 60
    print(f"✓ Done in {elapsed_min:.1f} min")

    # Save raw
    out_csv = RESULTS / "ablation_no_double.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "k", "traffic", "ir", "seed",
                                          "avg_lat", "status"])
        w.writeheader()
        for r in results:
            r["avg_lat"] = round(r["avg_lat"], 4) if np.isfinite(r["avg_lat"]) else "inf"
            w.writerow(r)
    print(f"✓ Per-run → {out_csv}")

    # Summary
    by_cell: dict = {}
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["traffic"], r["ir"])
        by_cell.setdefault(key, {}).setdefault(r["variant"], []).append(r["avg_lat"])

    print()
    print("=== DDDQN_full (Double DQN) vs DDDQN_no_double (Vanilla DQN target) ===")
    print(f"{'traffic':<10} | {'ir':>6} | {'full':>9} | {'no_double':>10} | {'Δ%':>7} | win?")
    print("-" * 65)
    summary_csv = RESULTS / "ablation_summary.csv"
    n_full_wins = 0
    n_total = 0
    rows = []
    for t in TRAFFIC:
        for ir in IR_LIST:
            cell = by_cell.get((t, ir), {})
            full = cell.get("DDDQN_full", [])
            nod = cell.get("DDDQN_no_double", [])
            if not (full and nod):
                continue
            full_m, nod_m = float(np.mean(full)), float(np.mean(nod))
            pct = 100 * (full_m - nod_m) / nod_m if nod_m else 0
            win = "yes" if full_m < nod_m else "no"
            if win == "yes":
                n_full_wins += 1
            n_total += 1
            print(f"{t:<10} | {ir:>6.3f} | {full_m:>9.2f} | {nod_m:>10.2f} | {pct:>+6.2f}% | {win}")
            rows.append(dict(traffic=t, ir=ir,
                             full_mean=round(full_m, 4),
                             no_double_mean=round(nod_m, 4),
                             delta_pct=round(pct, 2), full_wins=win))
    print(f"\nDDDQN-full beats no-Double in {n_full_wins}/{n_total} cells")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["traffic", "ir", "full_mean",
                                          "no_double_mean", "delta_pct", "full_wins"])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Summary → {summary_csv}")


if __name__ == "__main__":
    main()

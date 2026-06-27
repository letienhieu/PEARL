"""MC-2: Multi-seed training variance evaluation.

Evaluates 3 DDDQN models trained with different seeds (1, 42, 1337) on the
8x8 mesh ablation matrix per topic spec. Reports cross-seed variance to
address reviewer's MC-2 concern about single-seed training.
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

WEIGHTS = {
    "seed_1":    str(BUILD / "exports/ddqn_credit_aware_4x4_500ep.txt"),
    "seed_42":   str(BUILD / "exports/ddqn_credit_aware_seed42_500ep.txt"),
    "seed_1337": str(BUILD / "exports/ddqn_credit_aware_seed1337_500ep.txt"),
}
TRAFFIC = ["uniform", "hotspot", "transpose"]
IR_LIST = [0.005, 0.01, 0.02, 0.04, 0.08]
EVAL_SEEDS = [42, 1337, 2024, 314, 17]


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
    train_seed, k, traffic, ir, eval_seed = args
    weights = WEIGHTS[train_seed]
    cfg = make_config(k, traffic, ir, eval_seed)
    cfg_path = f"/tmp/mseed_{train_seed}_{k}_{traffic}_{ir}_{eval_seed}.cfg"
    Path(cfg_path).write_text(cfg)
    env = os.environ.copy()
    env["DDDQN_WEIGHTS"] = weights
    try:
        res = subprocess.run([BOOKSIM, cfg_path], cwd=str(BUILD), capture_output=True,
                             text=True, timeout=180, env=env)
        out = res.stdout
    except subprocess.TimeoutExpired:
        return dict(train_seed=train_seed, k=k, traffic=traffic, ir=ir,
                    eval_seed=eval_seed, avg_lat=float("inf"), status="timeout")
    m = re.findall(r"Packet latency average\s*=\s*([\d.]+)", out)
    avg_lat = float(m[-1]) if m else float("inf")
    return dict(train_seed=train_seed, k=k, traffic=traffic, ir=ir,
                eval_seed=eval_seed, avg_lat=avg_lat,
                status="ok" if np.isfinite(avg_lat) else "no_output")


def main():
    configs = [(ts, 8, t, ir, es)
               for ts in WEIGHTS
               for t in TRAFFIC
               for ir in IR_LIST
               for es in EVAL_SEEDS]
    print(f"Total runs: {len(configs)} (3 train seeds × 8x8 × 3 traffic × 5 IR × 5 eval seeds)")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(8) as pool:
        results = list(pool.imap_unordered(run_one, configs))
    print(f"✓ Done in {(time.time()-t0)/60:.1f} min")

    out_csv = RESULTS / "multi_seed_8x8.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_seed", "k", "traffic", "ir",
                                          "eval_seed", "avg_lat", "status"])
        w.writeheader()
        for r in results:
            r["avg_lat"] = round(r["avg_lat"], 4) if np.isfinite(r["avg_lat"]) else "inf"
            w.writerow(r)
    print(f"✓ Per-run CSV → {out_csv}")

    # Aggregate by (train_seed, traffic, ir): mean over 5 eval seeds
    agg = {}
    for r in results:
        if r["status"] != "ok": continue
        key = (r["train_seed"], r["traffic"], r["ir"])
        agg.setdefault(key, []).append(r["avg_lat"])

    # Per-train-seed mean per cell
    print()
    print("=== Per-cell mean latency by training seed (8x8 mesh) ===")
    print(f"{'traffic':<10} {'ir':>6} | {'seed 1':>9} {'seed 42':>9} {'seed 1337':>9} | {'cross_std':>10} {'cv%':>5}")
    print("-" * 75)
    cross_stats = []
    for t in TRAFFIC:
        for ir in IR_LIST:
            vals = []
            for ts in WEIGHTS:
                cell = agg.get((ts, t, ir), [])
                vals.append(np.mean(cell) if cell else float("nan"))
            v_arr = np.asarray(vals)
            std = float(np.nanstd(v_arr, ddof=1))
            mean = float(np.nanmean(v_arr))
            cv = 100 * std / mean if mean > 0 else 0
            cross_stats.append((t, ir, vals, std, mean, cv))
            print(f"{t:<10} {ir:>6.3f} | {vals[0]:>9.2f} {vals[1]:>9.2f} {vals[2]:>9.2f} | {std:>10.2f} {cv:>4.1f}%")

    print()
    cv_arr = np.asarray([cv for _,_,_,_,_,cv in cross_stats])
    print(f"Mean cross-seed CV: {cv_arr.mean():.2f}%")
    print(f"Max cross-seed CV: {cv_arr.max():.2f}%  (worst case)")

    # Save aggregated to CSV
    out_agg = RESULTS / "multi_seed_8x8_aggregated.csv"
    with open(out_agg, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traffic", "ir", "seed_1", "seed_42", "seed_1337",
                    "cross_std", "cross_mean", "cv_pct"])
        for t, ir, vals, std, mean, cv in cross_stats:
            w.writerow([t, f"{ir:.3f}", round(vals[0],4), round(vals[1],4),
                        round(vals[2],4), round(std,4), round(mean,4), round(cv,2)])
    print(f"✓ Aggregated → {out_agg}")


if __name__ == "__main__":
    main()

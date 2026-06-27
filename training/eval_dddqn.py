"""Evaluation matrix runner for DDDQN paper.

Sweeps: 6 methods × 3 meshes (4×4, 8×8, 16×16) × 3 traffic × 5 IRs × 3 seeds
      = 810 BookSim runs, multiprocessing.Pool(8).

Outputs:
  results_dddqn/per_run_metrics.csv   — raw single-run latencies
  results_dddqn/aggregated_metrics.csv — (method, k, traffic, ir) → mean, std, ci95
  results_dddqn/summary_dddqn_vs_minadapt.csv — paper headline table
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "booksim_build"
BOOKSIM = str(BUILD / "booksim")
RESULTS = ROOT / "results_dddqn"
DDDQN_WEIGHTS = str(BUILD / "exports/ddqn_4x4_retrained.txt")  # v2_500ep operative

METHODS = {
    # BookSim auto-appends "_mesh" to routing function name when topology=mesh,
    # so register the base name here (e.g. "odd_even" → resolves to odd_even_mesh).
    "XY":       "dor",
    "OddEven":  "odd_even",
    "DyAD":     "dyad",
    "MinAdapt": "min_adapt",
    "QRouting": "q_routing",
    "DDDQN":    "dddqn",
    # MC-1 ablation: credit-aware Min-Adaptive (rule-based, no DRL) — isolates
    # the contribution of the credit-aware sub-policy from the DRL agent.
    "CreditMinAdapt": "credit_aware_min_adapt",
}
MESHES = [4, 8, 16]
TRAFFIC = ["uniform", "hotspot", "transpose"]
IR_LIST = [0.005, 0.01, 0.02, 0.04, 0.08]
SEEDS = [42, 1337, 2024, 314, 17]


def make_config(k: int, traffic: str, ir: float, seed: int, routing_fn: str) -> str:
    return f"""topology=mesh; k={k}; n=2;
routing_function={routing_fn};
num_vcs=4; vc_buf_size=8; wait_for_tail_credit=1;
vc_allocator=islip; sw_allocator=islip; alloc_iters=1;
credit_delay=2; routing_delay=0; vc_alloc_delay=1; sw_alloc_delay=1;
input_speedup=2; output_speedup=1;
traffic={traffic}; packet_size=20; sim_type=latency;
injection_rate={ir}; sample_period=2000; warmup_periods=2; max_samples=2;
seed={seed};
"""


def _last_float(pat: str, text: str) -> float:
    m = re.findall(pat, text)
    return float(m[-1]) if m else float("nan")


def run_eval(args):
    method, k, traffic, ir, seed = args
    routing_fn = METHODS[method]
    cfg = make_config(k, traffic, ir, seed, routing_fn)
    cfg_path = f"/tmp/eval_{method}_{k}_{traffic}_{ir}_{seed}.cfg"
    Path(cfg_path).write_text(cfg)
    env = os.environ.copy()
    if method == "DDDQN":
        env["DDDQN_WEIGHTS"] = DDDQN_WEIGHTS
    timeout_s = {4: 60, 8: 180, 16: 600}[k]
    try:
        res = subprocess.run([BOOKSIM, cfg_path], cwd=str(BUILD), capture_output=True,
                             text=True, timeout=timeout_s, env=env)
        out = res.stdout
    except subprocess.TimeoutExpired:
        return dict(method=method, k=k, traffic=traffic, ir=ir, seed=seed,
                    avg_lat=float("inf"), net_lat=float("inf"), flit_lat=float("inf"),
                    accepted_pkt_rate=0.0, accepted_flit_rate=0.0,
                    status="timeout")
    avg_lat = _last_float(r"Packet latency average\s*=\s*([\d.]+)", out)
    net_lat = _last_float(r"Network latency average\s*=\s*([\d.]+)", out)
    flit_lat = _last_float(r"Flit latency average\s*=\s*([\d.]+)", out)
    pkt_rate = _last_float(r"Accepted packet rate average\s*=\s*([\d.]+)", out)
    flit_rate = _last_float(r"Accepted flit rate average\s*=\s*([\d.]+)", out)
    status = "ok" if np.isfinite(avg_lat) else "no_output"
    return dict(method=method, k=k, traffic=traffic, ir=ir, seed=seed,
                avg_lat=avg_lat, net_lat=net_lat, flit_lat=flit_lat,
                accepted_pkt_rate=pkt_rate, accepted_flit_rate=flit_rate,
                status=status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: 1 method × 1 mesh × 1 traffic × 1 seed")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)

    if args.smoke:
        configs = [("DDDQN", 4, "uniform", 0.02, 42)]
    else:
        configs = [(m, k, t, ir, s)
                   for m in METHODS
                   for k in MESHES
                   for t in TRAFFIC
                   for ir in IR_LIST
                   for s in SEEDS]
    n_total = len(configs)
    print(f"Total runs: {n_total} ({len(METHODS)} methods × {len(MESHES)} meshes × "
          f"{len(TRAFFIC)} traffic × {len(IR_LIST)} IR × {len(SEEDS)} seeds)")
    if not Path(BOOKSIM).exists():
        sys.exit(f"booksim binary not found: {BOOKSIM} — run `make` in booksim_build")

    ctx = mp.get_context("spawn")
    t0 = time.time()
    results = []
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_eval, configs)):
            results.append(r)
            done = i + 1
            if done % 20 == 0 or done == n_total:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_total - done)
                print(f"  [{done:4d}/{n_total}] elapsed={elapsed/60:.1f}m  "
                      f"eta={eta/60:.1f}m  last={r['method']:<8} k={r['k']:>2} "
                      f"{r['traffic']:<9} ir={r['ir']:.3f} s={r['seed']} → "
                      f"lat={r['avg_lat']:.2f} ({r['status']})", flush=True)

    elapsed_min = (time.time() - t0) / 60
    print(f"\n✓ All runs done in {elapsed_min:.1f} min")

    # Save per-run CSV
    csv_path = RESULTS / "per_run_metrics.csv"
    fieldnames = ["method", "k", "traffic", "ir", "seed",
                  "avg_lat", "net_lat", "flit_lat",
                  "accepted_pkt_rate", "accepted_flit_rate", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            r_copy = {k_: r.get(k_) for k_ in fieldnames}
            for fld in ("avg_lat", "net_lat", "flit_lat"):
                v = r_copy.get(fld)
                r_copy[fld] = round(v, 4) if v is not None and np.isfinite(v) else "inf"
            for fld in ("accepted_pkt_rate", "accepted_flit_rate"):
                v = r_copy.get(fld)
                r_copy[fld] = round(v, 6) if v is not None and np.isfinite(v) else 0.0
            w.writerow(r_copy)
    print(f"✓ Per-run CSV → {csv_path}")

    # Aggregate by (method, k, traffic, ir)
    agg: dict[tuple, list[float]] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["method"], r["k"], r["traffic"], r["ir"])
        agg.setdefault(key, []).append(r["avg_lat"])

    agg_path = RESULTS / "aggregated_metrics.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "k", "traffic", "ir",
                                          "mean", "std", "n", "ci95"])
        w.writeheader()
        for (m, k, t, ir), lats in sorted(agg.items()):
            arr = np.asarray(lats)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            w.writerow({"method": m, "k": k, "traffic": t, "ir": ir,
                        "mean": round(mean, 4), "std": round(std, 4),
                        "n": len(arr), "ci95": round(ci95, 4)})
    print(f"✓ Aggregated CSV → {agg_path}")

    # Summary table: DDDQN vs MinAdapt
    summary_path = RESULTS / "summary_dddqn_vs_minadapt.csv"
    print("\n=== DDDQN vs MinAdapt summary (% latency reduction) ===")
    print(f"{'k':>3} | {'traffic':<10} | {'ir':>6} | {'DDDQN':>9} | {'MinAdapt':>9} | "
          f"{'Δ%':>7} | {'win?':>5}")
    print("-" * 68)
    rows = []
    n_wins = 0
    n_total_compare = 0
    for k in MESHES:
        for t in TRAFFIC:
            for ir in IR_LIST:
                d = agg.get(("DDDQN", k, t, ir), [])
                m = agg.get(("MinAdapt", k, t, ir), [])
                if not (d and m):
                    continue
                dm = float(np.mean(d))
                mm = float(np.mean(m))
                pct = 100 * (dm - mm) / mm if mm else 0
                win = "yes" if dm < mm else "no"
                if dm < mm:
                    n_wins += 1
                n_total_compare += 1
                rows.append(dict(k=k, traffic=t, ir=ir, dddqn_mean=round(dm, 4),
                                 minadapt_mean=round(mm, 4), delta_pct=round(pct, 2),
                                 win=win))
                print(f"{k:>3} | {t:<10} | {ir:>6.3f} | {dm:>9.2f} | {mm:>9.2f} | "
                      f"{pct:>+6.2f}% | {win:>5}")
    win_rate = 100 * n_wins / n_total_compare if n_total_compare else 0
    print(f"\nDDDQN win rate over MinAdapt: {n_wins}/{n_total_compare} = {win_rate:.1f}%")

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "traffic", "ir", "dddqn_mean",
                                          "minadapt_mean", "delta_pct", "win"])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Summary CSV → {summary_path}")


if __name__ == "__main__":
    main()

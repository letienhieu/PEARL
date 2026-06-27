"""DDDQN training driver — BookSim subprocess + multiprocessing.Pool(8).

Usage:
    python3 train_dddqn.py --episodes 16 --workers 8        # smoke
    python3 train_dddqn.py --episodes 200 --workers 8       # full

Pipeline per chunk:
    1. Export current policy weights to /tmp/dddqn_weights_current.txt
    2. Pool.map: each worker runs `booksim` with DDDQN_EPSILON, DDDQN_LOG, DDDQN_WEIGHTS
    3. Parse per-episode log → trajectories grouped by packet_id → (s, a, r, s', done)
    4. Push transitions into shared replay buffer
    5. 20 Bellman updates with Double-DQN target + soft target update (tau=0.005)
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dqn_policy import DuelingDQN, ReplayBuffer  # noqa: E402

BUILD = ROOT / "booksim_build"
BOOKSIM = str(BUILD / "booksim")
WEIGHTS_CURRENT = "/tmp/dddqn_weights_current.txt"
EXPORT_DIR = BUILD / "exports"

TRAFFIC = ["uniform", "hotspot", "transpose"]
IR_LIST = [0.005, 0.01, 0.02, 0.04]


# ---------------------------------------------------------------------------
# BookSim config + episode runner
# ---------------------------------------------------------------------------

def make_config(k: int, traffic: str, ir: float, seed: int) -> str:
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


def run_episode(args):
    ep_id, k, traffic, ir, seed, eps = args
    cfg = make_config(k, traffic, ir, seed)
    cfg_path = f"/tmp/dddqn_cfg_ep_{ep_id}.cfg"
    log_path = f"/tmp/dddqn_log_ep_{ep_id}.txt"
    Path(cfg_path).write_text(cfg)
    env = os.environ.copy()
    env["DDDQN_WEIGHTS"] = WEIGHTS_CURRENT
    env["DDDQN_LOG"] = log_path
    env["DDDQN_EPSILON"] = f"{eps:.4f}"
    try:
        res = subprocess.run(
            [BOOKSIM, cfg_path],
            cwd=str(BUILD),
            capture_output=True, text=True, timeout=120, env=env,
        )
    except subprocess.TimeoutExpired:
        return float("inf"), log_path, traffic, ir
    m = re.search(r"Packet latency average\s*=\s*([\d.]+)", res.stdout)
    lat = float(m.group(1)) if m else float("inf")
    return lat, log_path, traffic, ir


# ---------------------------------------------------------------------------
# Log → trajectory parser
# ---------------------------------------------------------------------------

# Port indices: 0=E, 1=W, 2=N, 3=S — must match dddqn_mesh in routefunc.cpp
PORT_E, PORT_W, PORT_N, PORT_S = 0, 1, 2, 3


def chosen_port_from_state_action(state: np.ndarray, action: int) -> int:
    """Reconstruct the output port chosen by dddqn_mesh given (state, action).

    Mirrors C++ logic in routefunc.cpp:1016-1052:
      action 0 (XY)     : X first; if no X needed → Y.
      action 1 (adapt)  : credit-aware — among productive ports, pick the one
                         with most downstream space. Tie-break: Y-first.
    state[4] = ddx/k, state[5] = ddy/k (signed normalized destination delta).
    state[10..13] = downstream credit availability (1 − used/maxcredit),
                    higher = more space = less congested.
    """
    ddx = float(state[4])
    ddy = float(state[5])
    eps = 1e-6
    x_needed = abs(ddx) > eps
    y_needed = abs(ddy) > eps
    port_x = (PORT_E if ddx > 0 else PORT_W) if x_needed else -1
    port_y = (PORT_N if ddy > 0 else PORT_S) if y_needed else -1

    if action == 0:  # XY: X first, fall through to Y
        if port_x >= 0:
            return port_x
        return port_y if port_y >= 0 else PORT_E

    # action 1: credit-aware adaptive
    if port_x < 0 and port_y < 0:
        return PORT_E  # at dest (shouldn't reach here)
    if port_x < 0:
        return port_y
    if port_y < 0:
        return port_x
    # Both productive — pick port with higher credit availability
    # (equivalent to C++ "credit_x < credit_y" since avail = 1 - used/kMax).
    avail_x = float(state[10 + port_x])
    avail_y = float(state[10 + port_y])
    return port_x if avail_x > avail_y else port_y


def parse_log(log_path: str, final_latency: float, hotspot_bonus: bool = True):
    """Group decisions by packet_id, build (s, a, r, s', done) tuples.

    Reward shaping v3 (hotspot-aware):
        r_hop  = -beta - alpha · occ[chosen_port]         (queue pressure)
                 + bonus · 1[action=1 ∧ congested ∧ chose less-congested port]
        r_term = -gamma · packet_lat + delta · 1[budget]  (per-packet)
        packet_lat = step_last - step_first  (from log)
        budget     = 4 × n_logged_hops

    hotspot_bonus=True activates the +0.05 reward when in congested state
    (s[19] > 0.5) the agent picks adaptive (action 1) and the chosen port has
    occupancy below the max, i.e. genuine load-balancing behavior.
    """
    by_pid: dict[int, list[tuple[int, np.ndarray, int]]] = defaultdict(list)
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 27:
                continue
            try:
                step = int(parts[0])
                pid = int(parts[1])
                action = int(parts[4])
                state = np.asarray([float(x) for x in parts[7:27]], dtype=np.float32)
            except ValueError:
                continue
            by_pid[pid].append((step, state, action))

    # final_latency unused in v2 reward (per-packet only); kept in signature for caller compat
    _ = final_latency
    transitions = []
    alpha = 0.1      # queue-pressure coefficient
    beta = 0.01      # per-hop step penalty
    gamma = 0.02     # per-packet terminal latency coefficient
    delta = 0.5      # within-budget delivery bonus
    bonus_h = 0.05   # hotspot load-balancing bonus (v3)
    cong_thresh = 0.5  # s[19] threshold to flag a congested decision context
    for pid, hops in by_pid.items():
        hops.sort(key=lambda h: h[0])
        n = len(hops)
        if n == 0:
            continue
        packet_lat = (hops[-1][0] - hops[0][0]) if n > 1 else 0
        budget = 4 * n
        within = packet_lat <= budget
        for i, (_, s, a) in enumerate(hops):
            port = chosen_port_from_state_action(s, a)
            queue_press = float(s[6 + port])
            r = -beta - alpha * queue_press
            if hotspot_bonus and a == 1:
                max_occ = float(s[19])
                if max_occ > cong_thresh and queue_press < max_occ - 1e-3:
                    r += bonus_h  # adaptive avoided the most-congested port
            done = (i == n - 1)
            ns = hops[i + 1][1] if not done else np.zeros_like(s)
            if done:
                r += -gamma * packet_lat + (delta if within else 0.0)
            transitions.append((s, a, np.float32(r), ns, np.float32(done)))
    return transitions


# ---------------------------------------------------------------------------
# Weight exporter — must match C++ adapter block order exactly
# ---------------------------------------------------------------------------

WEIGHT_KEYS = [
    ("feature.0.weight",          (128, 20)),
    ("feature.0.bias",            (128,)),
    ("value_stream.0.weight",     (64, 128)),
    ("value_stream.0.bias",       (64,)),
    ("value_stream.2.weight",     (1, 64)),
    ("value_stream.2.bias",       (1,)),
    ("advantage_stream.0.weight", (64, 128)),
    ("advantage_stream.0.bias",   (64,)),
    ("advantage_stream.2.weight", (2, 64)),
    ("advantage_stream.2.bias",   (2,)),
]


def export_weights(model: nn.Module, path: str) -> None:
    sd = model.state_dict()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("# DDDQN weights — exported by train_dddqn.py\n")
        f.write("# layer_order: feature.0 (W,b), value_stream.0/2 (W,b), advantage_stream.0/2 (W,b)\n")
        for key, shape in WEIGHT_KEYS:
            v = sd[key].detach().cpu().numpy().astype(np.float32).flatten()
            assert v.size == int(np.prod(shape)), f"{key} size {v.size} != {np.prod(shape)}"
            f.write(f"# {key} shape={tuple(shape)} count={v.size}\n")
            for i in range(0, v.size, 8):
                f.write(" ".join(f"{x:.6e}" for x in v[i:i + 8]) + "\n")
    os.replace(tmp, path)


def import_weights(model: nn.Module, path: str) -> None:
    """Inverse of export_weights — load a 10-block text file back into model state_dict."""
    floats: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            floats.extend(float(x) for x in line.split())
    arr = np.asarray(floats, dtype=np.float32)
    expected = sum(int(np.prod(shape)) for _, shape in WEIGHT_KEYS)
    assert arr.size == expected, f"weight file has {arr.size} floats, expected {expected}"
    sd = model.state_dict()
    pos = 0
    for key, shape in WEIGHT_KEYS:
        n = int(np.prod(shape))
        sd[key] = torch.from_numpy(arr[pos:pos + n].reshape(shape).copy())
        pos += n
    model.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(n_episodes: int, n_workers: int = 8, seed_base: int = 1, save_every: int = 64,
          init_weights: str | None = None, epsilon_start: float = 1.0,
          history_path: str | None = None, mesh_sizes: list[int] | None = None,
          curriculum: str = "uniform", no_double: bool = False):
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)
    random.seed(seed_base)

    policy = DuelingDQN(state_dim=20, action_dim=2)
    target = DuelingDQN(state_dim=20, action_dim=2)
    if init_weights:
        import_weights(policy, init_weights)
        print(f"✓ Loaded init weights from {init_weights}")
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=1e-4)
    replay = ReplayBuffer(capacity=50_000)
    history = []

    EXPORT_DIR.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")

    eps_min = 0.1
    eps_span = max(epsilon_start - eps_min, 0.0)
    mesh_pool = mesh_sizes if mesh_sizes else [4]

    # Curriculum sampling weights for (TRAFFIC, IR_LIST).
    # "uniform"  : equal probability across all options (default).
    # "hotspot"  : oversample hotspot traffic + low IR where DDDQN was weakest
    #              (per Phase-1 curriculum-fix experiment).
    if curriculum == "hotspot":
        # TRAFFIC = [uniform, hotspot, transpose]
        traffic_w = [0.25, 0.50, 0.25]
        # IR_LIST = [0.005, 0.01, 0.02, 0.04] — heavier on low IRs (DDDQN was weakest there)
        ir_w = [0.30, 0.30, 0.20, 0.20]
    else:
        traffic_w = [1.0 / len(TRAFFIC)] * len(TRAFFIC)
        ir_w = [1.0 / len(IR_LIST)] * len(IR_LIST)
    print(f"  Training on mesh sizes: {mesh_pool}  curriculum={curriculum}")

    chunks = list(range(0, n_episodes, n_workers))
    for chunk_idx, chunk in enumerate(chunks):
        export_weights(policy, WEIGHTS_CURRENT)
        progress = chunk / max(n_episodes, 1)
        eps = max(eps_min, epsilon_start - progress * eps_span)
        rng = random.Random(seed_base + chunk)
        configs = []
        for w in range(n_workers):
            ep_id = chunk + w
            if ep_id >= n_episodes:
                break
            configs.append((
                ep_id, rng.choice(mesh_pool),
                rng.choices(TRAFFIC, weights=traffic_w, k=1)[0],
                rng.choices(IR_LIST, weights=ir_w, k=1)[0],
                seed_base + ep_id + 1,
                eps,
            ))

        with ctx.Pool(len(configs)) as pool:
            results = pool.map(run_episode, configs)

        n_added = 0
        for lat, log, _, _ in results:
            transitions = parse_log(log, lat)
            for t in transitions:
                replay.push(*t)
            n_added += len(transitions)

        # Bellman updates
        n_updates = 20 if len(replay) >= 64 else 0
        loss_avg = 0.0
        for _ in range(n_updates):
            states, actions, rewards, next_states, dones = replay.sample(64)
            states_t = torch.from_numpy(states).float()
            actions_t = torch.from_numpy(np.asarray(actions)).long().unsqueeze(1)
            rewards_t = torch.from_numpy(np.asarray(rewards)).float()
            next_t = torch.from_numpy(next_states).float()
            dones_t = torch.from_numpy(np.asarray(dones)).float()
            with torch.no_grad():
                if no_double:
                    # Vanilla DQN target (ablation): max over target_net only
                    next_q = target(next_t).max(dim=1)[0]
                else:
                    # Double DQN: policy_net selects, target_net evaluates
                    next_actions = policy(next_t).argmax(dim=1, keepdim=True)
                    next_q = target(next_t).gather(1, next_actions).squeeze(1)
                target_q = rewards_t + 0.99 * next_q * (1.0 - dones_t)
            curr_q = policy(states_t).gather(1, actions_t).squeeze(1)
            loss = ((curr_q - target_q) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_avg += loss.item()
            # Soft target update
            with torch.no_grad():
                for tp, pp in zip(target.parameters(), policy.parameters()):
                    tp.data.mul_(1 - 0.005).add_(pp.data, alpha=0.005)
        if n_updates:
            loss_avg /= n_updates

        lats = [r[0] for r in results if np.isfinite(r[0])]
        avg_lat = float(np.mean(lats)) if lats else float("inf")
        history.append({
            "chunk_end": min(chunk + n_workers, n_episodes),
            "avg_lat": avg_lat,
            "eps": round(eps, 4),
            "buf": len(replay),
            "added": n_added,
            "loss": round(loss_avg, 6),
        })
        print(f"[{min(chunk + n_workers, n_episodes):4d}/{n_episodes}] "
              f"avg_lat={avg_lat:8.2f}  eps={eps:.3f}  buf={len(replay):5d}  "
              f"added={n_added:4d}  loss={loss_avg:.4f}", flush=True)

        # Periodic checkpoint (also writes at end)
        if (chunk_idx + 1) % max(1, save_every // n_workers) == 0:
            export_weights(policy, str(EXPORT_DIR / "ddqn_4x4_retrained.txt"))

    out = EXPORT_DIR / "ddqn_4x4_retrained.txt"
    export_weights(policy, str(out))
    print(f"\n✓ Final weights → {out}")

    hist_file = Path(history_path) if history_path else (EXPORT_DIR / "training_history.csv")
    with open(hist_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_end", "avg_lat", "eps", "buf", "added", "loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"✓ History → {hist_file}")
    return policy, history


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init-weights", type=str, default=None,
                   help="Path to .txt weights to warm-start from (e.g. exports/ddqn_4x4_retrained_v2_500ep.txt)")
    p.add_argument("--epsilon-start", type=float, default=1.0,
                   help="Starting epsilon for ε-greedy. Lower for warm-start (e.g. 0.3).")
    p.add_argument("--history-path", type=str, default=None,
                   help="Optional output path for training_history.csv")
    p.add_argument("--mesh-sizes", type=str, default="4",
                   help="Comma-separated mesh sizes to sample per episode (e.g. '8' or '8,16')")
    p.add_argument("--curriculum", type=str, default="uniform",
                   choices=["uniform", "hotspot"],
                   help="Sampling curriculum: 'uniform' (equal weights) or 'hotspot' "
                        "(oversample hotspot traffic + low IR for Phase-1 fix)")
    p.add_argument("--no-double", action="store_true",
                   help="Ablation: use vanilla DQN target (max over target_net) instead of Double DQN")
    args = p.parse_args()
    if not Path(BOOKSIM).exists():
        sys.exit(f"booksim binary not found at {BOOKSIM} — run `make` in booksim_build first")
    mesh_sizes_list = [int(x) for x in args.mesh_sizes.split(",")]
    train(args.episodes, args.workers, seed_base=args.seed,
          init_weights=args.init_weights, epsilon_start=args.epsilon_start,
          history_path=args.history_path, mesh_sizes=mesh_sizes_list,
          curriculum=args.curriculum, no_double=args.no_double)

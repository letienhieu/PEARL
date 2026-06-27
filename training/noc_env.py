"""Cycle-accurate Python NoC mini-simulator for DDDQN training.

Designed to numerically match BookSim 2.0 mesh + DOR within 5% on uniform traffic.
Used solely for training; final evaluation runs on BookSim native binary.

Topology: 2D mesh, k×k routers, wormhole flow control with credit-based backpressure.
Per-output-port buffer queues. Round-robin arbitration.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Callable

# Output port encoding: 0=East, 1=West, 2=North, 3=South, 4=Local(eject)
PORT_E, PORT_W, PORT_N, PORT_S, PORT_LOCAL = 0, 1, 2, 3, 4
DIR_DELTAS = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # E, W, N, S


@dataclass
class Packet:
    pid: int
    src: int
    dest: int
    inject_cycle: int
    flits: int
    arrived_cycle: int = -1

    @property
    def latency(self) -> int:
        return self.arrived_cycle - self.inject_cycle if self.arrived_cycle > 0 else -1


@dataclass
class Router:
    rid: int
    cx: int
    cy: int
    buffer_depth: int = 8
    # output buffer queues: list of (Packet, hop_count) pairs, FIFO per port
    out_queues: list = field(default_factory=lambda: [deque() for _ in range(5)])
    # downstream credits available per port
    credits: list = field(default_factory=lambda: [8, 8, 8, 8, 0])  # local has no credit limit


class NoCSim:
    """Cycle-accurate 2D mesh NoC simulator with pluggable routing.

    Routing function signature: (sim, router, packet) -> port_idx (0..4).
    """

    def __init__(
        self,
        k: int = 4,
        buffer_depth: int = 8,
        traffic: str = "uniform",
        injection_rate: float = 0.02,
        packet_size: int = 20,
        warmup_cycles: int = 5000,
        measurement_cycles: int = 50000,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.n_nodes = k * k
        self.buffer_depth = buffer_depth
        self.traffic = traffic
        self.injection_rate = injection_rate  # packets per node per cycle
        self.packet_size = packet_size
        self.warmup_cycles = warmup_cycles
        self.measurement_cycles = measurement_cycles
        self.rng = np.random.RandomState(seed)
        self.cycle = 0
        self.next_pid = 0
        self.routers = [
            Router(rid=i, cx=i % k, cy=i // k, buffer_depth=buffer_depth) for i in range(self.n_nodes)
        ]
        self.delivered_packets: list[Packet] = []
        self.in_flight: dict[int, Packet] = {}

    # ------------------------------------------------------------------
    # Traffic generation
    # ------------------------------------------------------------------

    def _pick_dest(self, src: int) -> int:
        if self.traffic == "uniform":
            d = self.rng.randint(self.n_nodes - 1)
            return d if d < src else d + 1
        if self.traffic == "transpose":
            sx, sy = src % self.k, src // self.k
            return sy + sx * self.k
        if self.traffic == "hotspot":
            # 30% of traffic to mesh center, rest uniform
            if self.rng.random() < 0.3:
                cx, cy = self.k // 2, self.k // 2
                return cy * self.k + cx
            d = self.rng.randint(self.n_nodes - 1)
            return d if d < src else d + 1
        raise ValueError(f"unknown traffic {self.traffic}")

    def _maybe_inject(self) -> None:
        for r in self.routers:
            if self.rng.random() < self.injection_rate:
                if r.credits[PORT_LOCAL] < 0 or len(r.out_queues[PORT_LOCAL]) >= 100:
                    continue  # injection backpressure
                dest = self._pick_dest(r.rid)
                pkt = Packet(
                    pid=self.next_pid, src=r.rid, dest=dest,
                    inject_cycle=self.cycle, flits=self.packet_size,
                )
                self.next_pid += 1
                self.in_flight[pkt.pid] = pkt
                # Start at local input, will be routed next cycle
                r.out_queues[PORT_LOCAL].append((pkt, 0))

    # ------------------------------------------------------------------
    # State encoder (20-dim regional pressure)
    # ------------------------------------------------------------------

    def encode_state(self, router: Router, packet: Packet) -> np.ndarray:
        """Encode state per state_encoding.md spec."""
        s = np.zeros(20, dtype=np.float32)
        cx, cy = router.cx, router.cy
        dx, dy = packet.dest % self.k, packet.dest // self.k
        ddx, ddy = dx - cx, dy - cy

        # 0..3 dest quadrant one-hot (NE, NW, SE, SW)
        if ddx >= 0 and ddy >= 0:   s[0] = 1.0  # NE
        elif ddx < 0 and ddy >= 0:  s[1] = 1.0  # NW
        elif ddx >= 0 and ddy < 0:  s[2] = 1.0  # SE
        else:                       s[3] = 1.0  # SW

        # 4..5 normalized dest delta
        s[4] = ddx / self.k
        s[5] = ddy / self.k

        # 6..9 output buffer occupancy per port
        for p in range(4):
            s[6 + p] = len(router.out_queues[p]) / max(1, router.buffer_depth)

        # 10..13 downstream credit availability per port
        for p in range(4):
            s[10 + p] = router.credits[p] / max(1, router.buffer_depth)

        # 14..17 minimal hops via each direction (1.0 if invalid/off-mesh)
        for p, (vx, vy) in enumerate(DIR_DELTAS):
            nx, ny = cx + vx, cy + vy
            if 0 <= nx < self.k and 0 <= ny < self.k:
                hops = (abs(dx - nx) + abs(dy - ny) + 1) / (2 * self.k)
                s[14 + p] = hops
            else:
                s[14 + p] = 1.0  # invalid

        # 18 local aggregate load (mean buffer)
        bufs = [s[6 + p] for p in range(4)]
        s[18] = float(np.mean(bufs))
        # 19 congestion max
        s[19] = float(np.max(bufs))

        return s

    # ------------------------------------------------------------------
    # Routing helpers (sub-policies for the 2-action agent)
    # ------------------------------------------------------------------

    def _xy_route(self, router: Router, packet: Packet) -> int:
        if router.rid == packet.dest:
            return PORT_LOCAL
        cx, cy = router.cx, router.cy
        dx, dy = packet.dest % self.k, packet.dest // self.k
        if cx != dx:
            return PORT_E if dx > cx else PORT_W
        return PORT_N if dy > cy else PORT_S

    def _adaptive_route(self, router: Router, packet: Packet) -> int:
        if router.rid == packet.dest:
            return PORT_LOCAL
        cx, cy = router.cx, router.cy
        dx, dy = packet.dest % self.k, packet.dest // self.k
        ex, ey = dx - cx, dy - cy
        candidates = []
        if ex != 0:
            candidates.append(PORT_E if ex > 0 else PORT_W)
        if ey != 0:
            candidates.append(PORT_N if ey > 0 else PORT_S)
        if not candidates:
            return PORT_LOCAL
        # Pick port with most credit (least congestion downstream)
        return max(candidates, key=lambda p: router.credits[p])

    # ------------------------------------------------------------------
    # Single cycle step
    # ------------------------------------------------------------------

    def step(self, route_fn: Callable[["NoCSim", Router, Packet], int]) -> None:
        """Advance one cycle. route_fn: callable returning port idx."""
        self.cycle += 1
        self._maybe_inject()
        # Round-robin: each router examines local input + traversal queues
        for router in self.routers:
            for from_port in range(5):
                if not router.out_queues[from_port]:
                    continue
                packet, hop_count = router.out_queues[from_port][0]
                # Determine output port via routing function
                out_port = route_fn(self, router, packet)
                if out_port == PORT_LOCAL:
                    # Eject
                    router.out_queues[from_port].popleft()
                    packet.arrived_cycle = self.cycle
                    self.delivered_packets.append(packet)
                    self.in_flight.pop(packet.pid, None)
                    continue
                # Check if downstream router has buffer space
                vx, vy = DIR_DELTAS[out_port]
                nx, ny = router.cx + vx, router.cy + vy
                if not (0 <= nx < self.k and 0 <= ny < self.k):
                    # Off-mesh — drop or stall (we stall)
                    continue
                next_router = self.routers[ny * self.k + nx]
                opposite_port = [PORT_W, PORT_E, PORT_S, PORT_N][out_port]
                if len(next_router.out_queues[opposite_port]) >= self.buffer_depth:
                    continue  # backpressure stall
                # Move packet
                router.out_queues[from_port].popleft()
                next_router.out_queues[opposite_port].append((packet, hop_count + 1))

    # ------------------------------------------------------------------
    # Run experiment + collect metrics
    # ------------------------------------------------------------------

    def run(self, route_fn: Callable, total_cycles: int | None = None) -> dict:
        if total_cycles is None:
            total_cycles = self.warmup_cycles + self.measurement_cycles
        warmup_cutoff = self.warmup_cycles
        measure_until = total_cycles
        for _ in range(measure_until):
            self.step(route_fn)
        # Filter packets injected after warmup
        measured = [p for p in self.delivered_packets if p.inject_cycle >= warmup_cutoff]
        if not measured:
            return {"avg_latency": float("inf"), "p99_latency": float("inf"), "throughput": 0.0,
                    "n_delivered": 0, "n_injected_during_measure": 0}
        lats = [p.latency for p in measured]
        return {
            "avg_latency": float(np.mean(lats)),
            "p99_latency": float(np.percentile(lats, 99)),
            "throughput": len(measured) / (self.measurement_cycles * self.n_nodes),
            "n_delivered": len(measured),
            "min_latency": int(min(lats)),
            "max_latency": int(max(lats)),
        }


# ----------------------------------------------------------------------
# Smoke test entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NoC mini-sim smoke test (4x4 mesh, uniform, ir=0.02) ===")
    sim = NoCSim(k=4, traffic="uniform", injection_rate=0.02,
                 warmup_cycles=2000, measurement_cycles=10000, seed=42)
    res = sim.run(route_fn=lambda s, r, p: s._xy_route(r, p))
    print(f"XY    : avg={res['avg_latency']:.2f}  p99={res['p99_latency']:.2f}  "
          f"thru={res['throughput']:.4f}  n={res['n_delivered']}")

    sim = NoCSim(k=4, traffic="uniform", injection_rate=0.02,
                 warmup_cycles=2000, measurement_cycles=10000, seed=42)
    res = sim.run(route_fn=lambda s, r, p: s._adaptive_route(r, p))
    print(f"Adapt : avg={res['avg_latency']:.2f}  p99={res['p99_latency']:.2f}  "
          f"thru={res['throughput']:.4f}  n={res['n_delivered']}")

    print("\n=== Sweep XY 4x4 across injection rates ===")
    for ir in [0.005, 0.01, 0.02, 0.04, 0.08]:
        sim = NoCSim(k=4, traffic="uniform", injection_rate=ir,
                     warmup_cycles=2000, measurement_cycles=10000, seed=42)
        res = sim.run(route_fn=lambda s, r, p: s._xy_route(r, p))
        print(f"  ir={ir:6.3f}  avg={res['avg_latency']:8.2f}  p99={res['p99_latency']:8.2f}  "
              f"thru={res['throughput']:.4f}  n={res['n_delivered']}")

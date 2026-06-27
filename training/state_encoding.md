# DDDQN State Encoding — 20-dim Regional Pressure

## Design principles

1. **Mesh-size invariant** — state derived from local + 1-hop info only, no absolute coordinates.
   This enables zero-shot transfer (4×4 train → 8×8/16×16 deploy).
2. **Decision-time available** — every component readable from BookSim Router internals at the
   moment routing function is called (no future info needed).
3. **Bounded range** — all components normalized to [0, 1] or [-1, 1] for stable training.
4. **No identity leakage** — no router_id, dest_id, packet_id; only relative quantities.

## State vector layout (indices 0..19)

```
┌─────────────────────────────────────────────────────────────┐
│ Idx 0..3   dest_quadrant_onehot   ∈ {0,1}^4                 │
│            One-hot: [NE, NW, SE, SW] relative to current    │
│                                                              │
│ Idx 4..5   dest_delta_normalized   ∈ [-1, 1]^2              │
│            (dx_norm, dy_norm) = ((dx-cx)/k, (dy-cy)/k)      │
│                                                              │
│ Idx 6..9   output_buffer_occupancy ∈ [0, 1]^4               │
│            [E, W, N, S] = used_credits[port] / buffer_size  │
│                                                              │
│ Idx 10..13 downstream_credit_avail ∈ [0, 1]^4               │
│            [E, W, N, S] = downstream_free_slots / buf_size  │
│                                                              │
│ Idx 14..17 minimal_hops_per_dir    ∈ [0, 1]^4               │
│            [E, W, N, S] = manhattan_to_dest_via_dir / k     │
│            Returns 1.0 if direction is invalid (off-mesh)   │
│                                                              │
│ Idx 18     local_aggregate_load    ∈ [0, 1]                 │
│            mean of buffer occupancy across 4 output ports   │
│                                                              │
│ Idx 19     congestion_max          ∈ [0, 1]                 │
│            max of buffer occupancy across 4 output ports    │
└─────────────────────────────────────────────────────────────┘
```

## Action space (2 actions)

| Action | Meaning |
|--------|---------|
| 0 | Choose **dimension-order routing (XY)** at this hop |
| 1 | Choose **minimal-adaptive routing** at this hop (fall through to `min_adapt_mesh` logic) |

The agent's only decision per hop is policy *selection* (XY vs adaptive). The actual port-level
routing decision is delegated to the chosen sub-policy. This keeps the action space small and
enables stable training while exposing the key trade-off (deterministic-deadlock-free vs.
adaptive-load-balancing).

## Reward function

Per-step reward at each hop:
```
r_hop = -α · queue_pressure_chosen_port - β · 1[hop_taken]
```
where `α=0.1`, `β=0.01` (small penalty to discourage random non-progress moves).

End-of-packet reward:
```
r_terminal = -γ · packet_total_latency_normalized + δ · 1[delivered_within_budget]
```
where `γ=1.0`, `δ=0.5`, budget = 4 × min_hops (delivered fast → bonus).

## Mesh-size invariance proof sketch

- Indices 0-5: relative dest only (deltas, quadrants) — no absolute coords.
- Indices 6-13: per-port local quantities, count of buffers/credits is mesh-size-invariant
  (router buffer depth fixed, e.g. 8).
- Indices 14-17: hops-per-direction normalized by mesh side length k → in [0, 1] regardless.
- Indices 18-19: scalar aggregates over 4 ports → mesh-size invariant.

→ Network input distribution does not shift across mesh sizes (assuming injection rates are
  matched to relative load, which they are in our experimental matrix).

## Reference for paper Methods section

> "We encode each routing decision as a 20-dimensional regional pressure vector capturing
> destination quadrant (4-dim one-hot), normalized delta (2-dim), per-port output buffer
> occupancy (4-dim), downstream credit availability (4-dim), minimal hop count via each
> direction (4-dim), and aggregate load statistics (2-dim). All components are bounded in
> [-1, 1] or [0, 1] and computed from local plus one-hop neighbor information only,
> guaranteeing the encoding is mesh-size invariant — a prerequisite for zero-shot transfer
> across 4×4, 8×8, and 16×16 mesh deployments."

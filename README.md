# PEARL

**Policy Extracted via Adaptive Reinforcement Learning**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258390.svg)](https://doi.org/10.5281/zenodo.22258390)

A credit-aware minimal-adaptive routing sub-policy for mesh Network-on-Chip,
*discovered* by a Dueling Double DQN agent and *deployed* as a deterministic
rule with zero runtime parameters.

> Paper: **PEARL: Credit-Aware Minimal-Adaptive Routing for Mesh NoC** —
> Tien-Hieu Le, Duy-Hieu Bui, Xuan-Tu Tran — *11th IEEE International Conference on
> Integrated Circuits, Design and Verification (ICDV 2026), Hanoi, Vietnam*.

## Central thesis

**The rule is the contribution; the DRL is the discovery tool.**

A D3QN agent trained on a 4×4 mesh converges to a simple rule:
> Among productive output ports (those reducing Manhattan distance to the
> destination), select the port with the most downstream space (lowest
> `used_credit`).

We then evaluate this rule *standalone* — no DRL inference at runtime — against
five classical baselines and the full D3QN agent. The rule beats Min-Adaptive
on 26/45 cells, with up to −79% latency on transpose traffic, and matches the
DRL agent everywhere except where ε-greedy noise actively hurts it.

## Headline results

- **PEARL vs Min-Adaptive**: 26/45 cells better (58%, paired *t*-test, p<0.05)
- **PEARL vs full D3QN agent**: tied on 28/45 cells (62%, |Δ|<0.5 cycle);
  PEARL wins 10/45, agent wins only 7/45 → 11K-param inference cost not
  worth the marginal gain
- **Largest gains vs Min-Adaptive**: −79% latency on 4×4 transpose ir=0.02,
  −69% on 8×8, −56% on 16×16
- **Zero-shot transfer**: trained on 4×4 only, deployed unchanged on
  8×8 and 16×16
- **Action distribution**: D3QN picks the credit-aware action on 98.4% of
  routing decisions across the 45-cell matrix → empirically validates the
  "rule is the contribution" framing
- **13/13 integration tests pass**, including C++↔Python forward equivalence
  and mesh-size invariance

## Repo structure

```
PEARL/
├── src/                  BookSim 2.0 source with PEARL + D3QN integration
│   ├── routefunc.cpp     7 routing functions (XY, OddEven, DyAD,
│   │                     MinAdapt, QRouting, D3QN, PEARL)
│   └── dqn_*.{cpp,py}    D3QN agent + Python ↔ C++ adapter
├── training/             Training, evaluation, statistics, figure scripts
│   ├── train_dddqn.py      D3QN training (500 episodes, ~2:12 on M1)
│   ├── eval_dddqn.py     Full evaluation matrix (1,575 BookSim runs)
│   └── make_figures.py   Publication-quality figures
├── tests/                13 integration tests
├── results_dddqn/        Aggregated CSVs + figures (from 5-seed runs)
```

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`

```bash
pip install -r requirements.txt
```

## Reproducing the results

The tables and figures in the paper are regenerated directly from the aggregated
5-seed data in `results_dddqn/`, without re-running the full 1,575-run matrix:

```bash
cd training
python3 make_figures.py
```

The routing implementation in `src/` builds on **BookSim 2.0**
(https://github.com/booksim/booksim2); the training and evaluation drivers in
`training/` invoke a compiled BookSim binary. A complete, self-contained
build-and-run environment accompanies the extended version of this work.

## Method summary

**Discovery (offline, training time)**: Dueling Double DQN
[Wang et al. 2016; van Hasselt et al. 2016]
- Architecture: feature(20 → 128) → V-stream(1) + A-stream(2). ~11K params.
- State: 20-dim mesh-size-invariant
  (quadrant + Δ + buffer + credit + hops + summary)
- Actions: 2 — XY (DOR) or credit-aware minimal-adaptive sub-policy
- Reward: per-hop queue pressure + per-packet terminal latency
- Training: 500 episodes on 4×4 mesh, Adam @ 1e-4, soft τ=0.005,
  ε linearly 1.0 → 0.1

**Deployment (online, runtime)**: PEARL rule, pure C++
- Among productive ports `P` (reducing Manhattan distance), pick
  `p* = argmin used_credit(p)`
- Zero trainable parameters at runtime
- O(P) per-decision cost, comparable to existing Min-Adaptive tie-break
- Uses existing `IQRouter::GetUsedCredit(port)` BookSim 2.0 interface

## Baselines

7 classical NoC routing functions implemented in BookSim 2.0:

| # | Method | Reference |
|---|---|---|
| 1 | XY / DOR | Dimension-Order Routing |
| 2 | OddEven | Chiu, IEEE TPDS 2000 |
| 3 | DyAD | Hu & Marculescu, DAC 2004 |
| 4 | Min-Adaptive | Duato 1993 |
| 5 | Q-routing HARAQ | Boyan & Littman, NIPS 1994 |
| 6 | D3QN (full DRL agent, discovery tool — this work) |  |
| 7 | **PEARL** (extracted credit-aware rule, deployable — this work) |  |

## Citation

```bibtex
@inproceedings{Le2026PEARL,
  author    = {Le, Tien-Hieu and Bui, Duy-Hieu and Tran, Xuan-Tu},
  title     = {{PEARL}: Credit-Aware Minimal-Adaptive Routing for Mesh {NoC}},
  booktitle = {Proc.\ 11th IEEE Int.\ Conf.\ on Integrated Circuits,
               Design and Verification (ICDV)},
  year      = {2026},
  address   = {Hanoi, Vietnam},
  month     = {Oct.},
}
```

## License

- Code: **MIT**
- Paper sources: **CC-BY 4.0**

## Acknowledgments

The authors thank the IEEE ICDV 2026 program committee and the BookSim 2.0
maintainers. All experiments were conducted on a single Apple M1 laptop.

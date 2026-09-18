# CarlaGym

<p align="center">
  <img src=".github/carla-logo.png" height="auto" width="150"> <br>
   A Gymnasium environment for Reinforcement Learning Agents in Congested SDNs
</p>


A modular network-simulation RL environment, architecturally inspired by
`mobile-env`: traffic generation, topology/queueing, and reward computation
are fully decoupled from the `gymnasium.Env` loop.

## Layout

```
carla_gym/
  env.py                  # CarlaGymEnv(gymnasium.Env) — spaces + step()/reset() only
  core/
    network.py            # Topology, node buffers, link service, store-and-forward
    traffic.py            # Poisson QoS traffic (voice/gaming/video/best-effort)
  utils/
    reward.py             # R = Throughput - k*Delay - c*PacketLoss (normalized)
    visualization.py      # Matplotlib congestion rendering
examples/
  random_agent_test.py    # Sprint 4: random + 3 reference policies, pandas logging
tests/
  test_env.py             # 19 tests: API conformance + simulator invariants
```

## Install & run

```bash
pip install -r requirements.txt

PYTHONPATH=. python -m pytest tests/ -q
PYTHONPATH=. python examples/random_agent_test.py --episodes 3 --steps 150
PYTHONPATH=. python examples/random_agent_test.py --budget 0.8   # constrained variant
```

## Simulation model

The environment is a **store-and-forward** packet simulator, not a
flow-rate approximation:

* **Nodes** hold finite buffers (units). Packets physically occupy them.
* **Links** have a service rate (units/step). The action scales that rate.
* Packets advance **at most one hop per step**, subject to link budget and
  downstream buffer space, with FIFO head-of-line blocking.
* **Latency is measured** as the number of steps from injection to
  delivery. It is not a utilization proxy.
* **Loss** occurs on admission overflow, downstream buffer overflow, or
  unroutable src/dst.

Routing is rigid shortest-path (Sprint 2), so Phase 1 isolates *resource
allocation*; learned routing is the Phase-2 extension.

### Spaces

| | Shape | Meaning |
|---|---|---|
| Observation | `Box(0,1,(2E+N,))` | link utilization (E), pending demand per link (E), node buffer occupancy (N) |
| Action | `Box(0,1,(E,))` | per-link bandwidth weight; service rate = `capacity × weight` |

### Reward

```
R = Throughput − k·Delay − c·PacketLoss
```

All three terms are **normalized to [0,1]** before the coefficients apply:
throughput is a goodput ratio, delay is measured latency over
`delay_reference`, loss is a drop ratio. Defaults `k=0.5`, `c=1.0`.

Delay is weighted by QoS class `delay_sensitivity`, so latency on
voice/gaming costs more than the same latency on best-effort.

## Phase 1 status

| Sprint | Deliverable | Status |
|---|---|---|
| 1 | Repo skeleton, env config, `CarlaGymEnv` | Done |
| 2 | 5 topologies, Poisson QoS traffic, shortest-path routing | Done |
| 3 | Observation/action spaces, wired QoS reward | Done |
| 4 | Random-agent run + baselines, pandas logs, sensitivity checks | Done |

### Sprint 4 results (grid, 3 episodes × 150 steps, post-warmup)

| Policy | Reward | Throughput | Loss | Latency (steps) |
|---|---|---|---|---|
| max_bandwidth | 0.524 | 0.832 | 0.088 | 3.47 |
| queue_proportional | 0.423 | 0.821 | 0.114 | 4.44 |
| random | −0.699 | 0.405 | 0.614 | 18.68 |
| min_bandwidth | −1.000 | 0.000 | 1.000 | — |

All six Sprint-4 assertions pass: node occupancy is a live observable,
actions move backlog and reward, the policy ranking is non-degenerate, and
both QoS terms stay informative rather than pinned at their bounds.

## Two calibration notes worth defending in the report

**1. Capacity must be tuned to the offered load.** Default traffic offers
~104 units/step. Too much capacity and packet loss sits at exactly 0, so
the loss term contributes nothing; too little and the network collapses to
100% loss, so every action looks equally bad. `link_capacity=30`,
`node_buffer=120` was chosen by sweeping to the regime where a good policy
reaches ~0.85 throughput and a bad one ~0.4 — a wide, learnable gap.

**2. Unconstrained bandwidth makes the optimal policy trivial.** With no
cost to allocation, "output 1.0 everywhere" is optimal and an agent learns
it almost immediately — an unimpressive result to present. Setting
`bandwidth_budget=0.8` caps total spend, and the ranking inverts:

| Policy | Unconstrained | `budget=0.8` |
|---|---|---|
| max_bandwidth | **0.537** | 0.086 |
| queue_proportional | 0.428 | **0.344** |

Under a budget the agent must decide *where* to spend, and
congestion-aware allocation beats uniform allocation four-fold. This is the
more defensible setting for demonstrating that RL adds value.

## Phase 2 directions

* **Learned routing** — fold routing weights into the action space and drop
  the `shortest_path` call in `network.path()`. The store-and-forward core
  already advances packets hop-by-hop, so per-hop routing decisions slot in
  without rewriting the simulator.
* **Constrained allocation** — adopt `bandwidth_budget` as the default and
  compare PPO/SAC against the `queue_proportional` heuristic as the baseline
  to beat.
* **Multi-agent** — `core/` is independent of the RL formulation, so a
  per-node controller setup only requires rewiring `env.py`.
* **Deadline-aware reward** — `deadline_misses` is already logged per step
  but not yet in the reward; it is a natural third QoS term.

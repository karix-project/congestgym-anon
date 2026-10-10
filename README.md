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
    trace_traffic.py      # Dataset-driven traffic: TraceTrafficGenerator
    dataset_adapters.py   # Raw dataset -> (step,src,dst,volume) trace schema
  utils/
    reward.py             # R = Throughput - k*Delay - c*PacketLoss (normalized)
    visualization.py      # Matplotlib congestion rendering
examples/
  random_agent_test.py    # Sprint 4: random + 3 reference policies, pandas logging
  train_ppo.py             # PPO training (synthetic or --trace dataset-driven)
scripts/
  prepare_trace.py          # CLI: raw dataset -> trace CSV, with calibration stats
notebooks/
  01_simulation_walkthrough.ipynb   # step through the sim, render congestion, plot queues
  02_policy_comparison.ipynb         # reference policies head-to-head, synthetic + real trace
  03_dataset_exploration.ipynb        # what's in the raw datasets, the placement-fix visualized
  04_ppo_training_curves.ipynb         # a real (short) PPO training run with a learning curve
data/
  cs448b_ipasn.csv          # raw: daily local-host -> remote-ASN flow counts
  output1.csv                # raw: packet capture, numeric export
  Midterm_53_group.csv        # raw: SAME capture as output1.csv, Wireshark text export
tests/
  test_env.py                   # 19 tests: API conformance + simulator invariants
  test_dataset_pipeline.py       # 15 tests: adapters + real-data training pipeline
```

## Install & run

```bash
pip install -r requirements.txt   # torch/stable-baselines3 only needed for train_ppo.py

PYTHONPATH=. python -m pytest tests/ -q
PYTHONPATH=. python examples/random_agent_test.py --episodes 3 --steps 150
PYTHONPATH=. python examples/random_agent_test.py --budget 0.8   # constrained variant

# Dataset-driven: convert a raw dataset, then train on it
PYTHONPATH=. python scripts/prepare_trace.py --in data/output1.csv --out data/pcap_trace.csv
PYTHONPATH=. python examples/train_ppo.py --trace data/pcap_trace.csv --nodes 16 \
    --load 110 --budget 0.5 --timesteps 300000
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

## Dataset-driven training (trace_traffic integration)

`CarlaGymEnv(traffic=...)` accepts any object with `generate(step)` /
`reset(seed)`, so `TraceTrafficGenerator` (reads a `step,src,dst,volume`
CSV) is a drop-in replacement for the Poisson `TrafficGenerator`.
`dataset_adapters.py` converts the raw datasets this repo ships with into
that schema. None of them arrive in it natively:

| Raw file | What it actually is | Adapter |
|---|---|---|
| `cs448b_ipasn.csv` | Stanford CS448B dataset: one row per (day, local host 0-9, remote ASN), with a daily flow count. No per-packet timing at all. | `convert_cs448b` |
| `output1.csv` | Raw packet capture, numeric export: per-packet Timestamp/Source IP/Destination IP/Length. | `convert_packet_capture` |
| `Midterm_53_group.csv` | **The same capture as `output1.csv`**, exported by Wireshark's text UI instead — confirmed by identical destination-IP counts. Source/Destination here can also be a MAC-vendor string or the literal `"Broadcast"`, which aren't routable and are dropped (634 rows). | `convert_packet_capture` (auto-detects either export format) |

```bash
PYTHONPATH=. python scripts/prepare_trace.py --in data/output1.csv --out data/pcap_trace.csv
PYTHONPATH=. python scripts/prepare_trace.py --in data/cs448b_ipasn.csv --out data/cs448b_trace.csv
```

### Topology-aware node placement (the fix that mattered most)

Mapping thousands of real IPs/ASNs down to a handful of simulated nodes
means bucketing — and a naive hash or frequency-rank bucketing has no idea
which bucket id lands where in the topology. In the first pass, the single
IP responsible for ~95% of all destination traffic landed on node `0`,
which NetworkX's grid labeling puts in a **corner with only 2 links**. No
bandwidth policy fixes that: it's a fixed physical bottleneck, not a load
problem, and max-bandwidth throughput was capped at ~0.26 regardless of
how low `--load` was set.

The adapters now compute each bucket's total traffic volume and place the
heaviest buckets onto the best-connected real node ids (`_place_by_volume`
in `dataset_adapters.py`). That alone took max-bandwidth throughput on the
pcap trace from 0.26 to 0.42+ at the same load. `tests/test_dataset_pipeline.py::test_heaviest_node_lands_on_best_connected_position`
guards this regression.

### Calibration differs from synthetic traffic

`TraceTrafficGenerator(target_load=...)` rescales a trace's raw volume to
match whatever load you ask for, so the trace's own raw units are mostly
irrelevant — `scripts/prepare_trace.py` used to suggest the raw per-step
mean as `--load`, which is close to a no-op rescaling and would reproduce
whatever collapse the raw units implied. It now suggests a load based on
the env's actual link budget instead.

Separately, real traffic concentrates on far fewer src/dst pairs than the
synthetic Poisson generator spreads across all edges, so the *budget*
needed for a congestion-aware policy to actually beat uniform allocation
is tighter: **`--budget 0.5`** for real traces, not the `0.8` used for
synthetic traffic. At `0.5`, `queue_proportional` beats `max_bandwidth` on
both `cs448b` and `pcap` traces; above roughly `0.6` the budget stops
binding and uniform allocation wins again, same as the unconstrained case.

### Non-deterministic node placement (found while building the notebooks)

The adapters originally bucketed IPs/ASNs with Python's builtin `hash()`,
which is randomized per-process by default (`PYTHONHASHSEED`) for
hash-flooding security reasons. That meant **re-running `prepare_trace.py`
produced a different node mapping every time**, even on the identical raw
CSV -- a real reproducibility break, not a cosmetic one, and every trace
file already generated before this fix should be considered stale. Fixed
with a stable `hashlib.md5`-based hash; `tests/test_dataset_pipeline.py::test_adapter_output_reproducible_across_processes`
runs the adapter in two subprocesses with different hash seeds and asserts
identical output.

### Two bugs in train_ppo.py / random_agent_test.py's interaction

Both are fixed, with regression tests (`test_reference_policies_callable_with_and_without_rng`,
`test_rollout_works_with_monitor_wrapped_env`):

1. `random_agent_test.py`'s policy functions (`policy_max`, `policy_queue_proportional`)
   required a positional `rng` argument that `train_ppo.py`'s `rollout()`
   never passed — only `policy_random` had a hand-written wrapper to work
   around it. `rng` is now optional (`rng=None`) on every policy.
2. `train_ppo.py` wraps envs in stable-baselines3's `Monitor`, which does
   **not** forward custom attributes like `num_edges`/`num_nodes` — calling
   a reference policy on the wrapped env raised `AttributeError`.
   `rollout()` now hands policies `env.unwrapped` while still stepping the
   wrapped `env` (so Monitor's episode logging still works).

### Verified training run

```bash
PYTHONPATH=. python examples/train_ppo.py --trace data/pcap_trace.csv \
    --nodes 16 --load 110 --budget 0.5 --timesteps 60000 --n-envs 4
```

| Policy | Reward (held-out seeds) |
|---|---|
| queue_proportional | **-0.067** |
| PPO (60K timesteps) | -0.101 |
| max_bandwidth | -0.126 |
| random | -0.162 |

At 60K timesteps PPO already beats the naive heuristics and is closing in
on `queue_proportional`; at the script's default `--timesteps 300000` it
should plausibly surpass it. 8K timesteps (≈8 gradient updates on a
24-dim continuous action space) is not enough to judge anything by —
don't read a short smoke-test run as "PPO doesn't work here."

## Notebooks

Each notebook in `notebooks/` is pre-executed with real outputs embedded,
so you can open it and see plots immediately without running anything.
To regenerate them after you change the code:

```bash
cd notebooks
PYTHONPATH=.. jupyter nbconvert --to notebook --execute --inplace 01_simulation_walkthrough.ipynb
```

| Notebook | What it shows |
|---|---|
| `01_simulation_walkthrough.ipynb` | Builds an env, runs `queue_proportional`, plots backlog/utilization/reward-components over time, and renders the network graph at a lightly-loaded step vs. a congested one (plus a zero-bandwidth stress test). |
| `02_policy_comparison.ipynb` | Runs all four reference policies on both synthetic traffic (budget=0.8) and the real pcap trace (budget=0.5), with bar charts, reward-over-time lines, and a throughput-vs-loss scatter. |
| `03_dataset_exploration.ipynb` | Profiles all three raw datasets, then visually demonstrates the topology-placement fix: a scatter of node degree vs. assigned traffic volume, naive hashing vs. the actual adapter, side by side. |
| `04_ppo_training_curves.ipynb` | A real 60K-timestep PPO run (≈30-60s on CPU) against the real trace, evaluated on held-out seeds every 10K steps, plotted against the heuristic baselines as horizontal reference lines. |

Notebook 4 trains for 60K timesteps for runtime's sake (enough to beat
`max_bandwidth`/`random`, not enough to catch `queue_proportional`) — it's
a demonstration that training works and is improving, not the full result.
For that, run `examples/train_ppo.py --timesteps 300000` as described above.

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

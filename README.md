# CarlaGym

<p align="center">
  <img src=".github/carla-logo.png" height="auto" width="150"> <br>
   A Gymnasium environment for reinforcement learning on congested network traffic
</p>

CarlaGym is a Python reinforcement-learning framework built on the standard
[Gymnasium](https://gymnasium.farama.org/) API. An agent controls a congested
packet network: every step it decides **how much bandwidth each link gets** and
**how traffic is routed**, and it is rewarded on QoS metrics (throughput, delay,
packet loss, deadline misses). The framework ships with customizable
topologies, volatile synthetic and dataset-driven traffic, node-bottleneck
scenarios, rule-based traffic-engineering baselines, and scripts to train,
evaluate and benchmark RL agents against them.

Traffic generation, the network simulator and the reward are decoupled from
the `gymnasium.Env` loop (an architecture inspired by `mobile-env`), so each
can be replaced independently.

## Layout

```
carla_gym/
  env.py                  # CarlaGymEnv(gymnasium.Env): spaces, step()/reset()
  marl_env.py             # MultiAgentCarlaGymEnv (PettingZoo ParallelEnv)
  baselines.py            # rule-based reference policies (bandwidth and routing-aware)
  core/
    network.py            # topologies, node buffers, link service, store-and-forward
    step.py               # one simulation step, shared by both envs
    traffic.py            # Poisson QoS traffic (voice / gaming / video / best-effort)
    trace_traffic.py      # dataset-driven traffic: TraceTrafficGenerator
    dataset_adapters.py   # raw dataset -> (step, src, dst, volume) trace
  utils/
    reward.py             # R = Throughput - k*Delay - c*PacketLoss - m*DeadlineMiss
    stats.py              # 95% confidence intervals
    visualization.py      # Matplotlib congestion rendering
examples/
  random_agent_test.py    # sanity run: all baselines, pandas log, six checks
  train_agent.py          # train PPO / SAC / TD3, held-out evaluation vs baselines
  train_ppo.py            # shortcut for train_agent.py --algo ppo
scripts/
  prepare_trace.py        # raw dataset -> trace CSV
  run_benchmark.py        # baseline scenario matrix with confidence intervals
  summarize_training.py   # aggregate multi-seed training runs
  update_readme_results.py  # copy result tables from results/ into this file
notebooks/                # four executed notebooks (see below)
data/                     # raw datasets and the traces derived from them
results/                  # outputs of the runs reported below
tests/                    # pytest suite
qa/                       # reproduction scripts and logs for the QA round
docs/                     # tester report and researcher note
```

## Install

Python 3.10 or newer.

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -c constraints.txt
$env:PYTHONPATH = "."
```

bash (Linux / macOS):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
export PYTHONPATH=.
```

`constraints.txt` pins the versions everything here was run with; drop
`-c constraints.txt` to take the newest compatible releases. As a package:
`pip install -e .` for the core environment, or `pip install -e ".[all]"` to add
the extras `rl` (PyTorch, Stable-Baselines3), `marl` (PettingZoo), `notebooks`
and `dev`.

The commands below are the same in both shells once `PYTHONPATH` is set.

```
python -m pytest -q
python examples/random_agent_test.py --episodes 3 --steps 150 --budget 0.8
python examples/train_agent.py --algo ppo --timesteps 500000 --seed 0
python scripts/run_benchmark.py --quick
```

## Quick start

```python
import gymnasium as gym
import carla_gym  # registers CarlaGym-v0

env = gym.make("CarlaGym-v0", topology="grid", num_nodes=9, bandwidth_budget=0.8)
obs, info = env.reset(seed=0)
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
```

Main constructor options: `topology` (`grid`, `ring`, `star`, `mesh`,
`random`), `num_nodes`, `link_capacity`, `node_buffer`, `load_scale`,
`bandwidth_budget`, `bottlenecks`, `traffic`, reward coefficients `k`, `c`,
`m`, `delay_reference`, `max_steps`, `render_mode` (`human` or `rgb_array`).

## Simulation model

The environment is a **store-and-forward** packet simulator, not a flow-rate
approximation.

* **Nodes** hold finite buffers. Packets physically occupy them.
* **Links** are undirected and have a service rate (units per step) shared by
  both directions. The bandwidth action scales that rate.
* Packets advance **at most one hop per step**. Service is **fractional**: a
  packet larger than the budget a link has left is partly transmitted and
  finishes the hop on a later step, so any positive allocation moves traffic.
* Each node forwards in FIFO order with **head-of-line blocking**.
* A packet moves on only if the next node had room at the start of the step
  (plus what has arrived since); otherwise it is dropped. Buffers therefore
  never exceed `node_buffer`.
* **Routing** is decided when a packet is admitted: Dijkstra shortest path
  under the per-link routing weights in the current action. Uniform weights
  give hop-count shortest paths.
* **Latency is measured** in steps from injection to delivery.
* **Loss** occurs on admission overflow, downstream buffer overflow, or when
  no route exists.

### Spaces

`E` = number of links, `N` = number of nodes (default grid: `E=12`, `N=9`).

| | Shape | Meaning |
|---|---|---|
| Observation | `Box(0, 1, (2E+N,))` | link utilization (E), pending demand per link (E), node buffer occupancy (N) |
| Action | `Box(0, 1, (2E,))` | per-link bandwidth weight (E), then per-link routing weight (E) |

`bandwidth_budget=b` caps the sum of bandwidth weights at `b*E`; actions above
it are scaled down proportionally. Without a budget, "allocate everything
everywhere" is close to optimal and the allocation problem is uninteresting.

### Reward

```
R = Throughput - k*Delay - c*PacketLoss - m*DeadlineMiss        (defaults k=0.5, c=1.0, m=0.5)
```

Every term is normalized to [0, 1] per step:

* **Throughput**: delivered / offered units, clipped at 1.
* **Delay**: delay-sensitivity-weighted mean age of traffic in the system
  (end-to-end delay of packets delivered this step, current age of packets
  still queued), divided by `delay_reference` (10 steps) and clipped at 1.
  It is QoS-weighted, not plain latency: voice and gaming cost more per step
  than best-effort.
* **PacketLoss**: dropped / offered units, clipped at 1.
* **DeadlineMiss**: packets that missed their class deadline this step
  (delivered late, or expired while queued) over packets resolved this step.
  Each packet is counted once.

The reward is therefore in [-2, 1]. `info` reports every term and the raw
counters.

### Traffic

`TrafficGenerator` draws Poisson arrivals for four QoS classes with different
packet sizes, delay sensitivities and deadlines (about 104 units per step at
`load_scale=1`). `TraceTrafficGenerator` replays a `step,src,dst,volume` CSV,
rescaled to a `target_load`. Any object with `generate(step)` and
`reset(seed)` can be passed as `traffic=`.

Seeding: `reset(seed=s)` replays the same traffic; `reset()` continues the
random stream, so successive episodes see new traffic.

### Node bottlenecks

`bottlenecks={node: scale}` degrades a node: it keeps only that fraction of
its buffer and of the capacity of every link touching it. This is the
scenario for studying bottleneck mitigation: a policy has to steer traffic
around the degraded node or prioritise its links.

### Multi-agent

`MultiAgentCarlaGymEnv` exposes the same simulator through the PettingZoo
parallel API. Each link is controlled by exactly one of its endpoint nodes;
nodes that control no link are not agents. Agents share the global
observation and a cooperative reward. It needs the `marl` extra.

## Baselines

All in `carla_gym/baselines.py`:

| Policy | Bandwidth | Routing |
|---|---|---|
| `max_bandwidth` | equal split over all links | hop-count shortest path |
| `queue_proportional` | proportional to each link's pending demand | hop-count shortest path |
| `load_aware_routing` | equal split | load-aware shortest path (congested links cost more) |
| `congestion_aware` | proportional to pending demand | load-aware shortest path |
| `random` | uniform random | uniform random |
| `min_bandwidth` | 5% everywhere | hop-count shortest path |

`load_aware_routing` and `congestion_aware` are the routing-aware
traffic-engineering baselines.

## Results

Everything in this section is copied from files under `results/` by
`scripts/update_readme_results.py`; the commands that produced them are shown.
Rewards are the mean per-step reward after a 20-step warm-up, over 200-step
episodes.

### RL agents vs baselines (3x3 grid, synthetic traffic, budget 0.8)

```
python examples/train_agent.py --algo ppo --timesteps 500000 --seed S --eval-every 25000 --out results/grid_b08/ppo_seedS
python examples/train_agent.py --algo sac --timesteps 100000 --seed S --eval-every 10000 --out results/grid_b08/sac_seedS
python scripts/summarize_training.py results/grid_b08
```

for `S` in 0..4. Agents train on a `[-1, 1]` action space
(`gymnasium.wrappers.RescaleAction`). Training envs draw fresh traffic every
episode, learning curves use validation seeds (1000+), and the table uses test
seeds (2000 to 2009) that neither training nor the curves saw. For RL agents
the interval is over the five training seeds; for baselines it is over the ten
test episodes.

<!-- RESULTS:training:start -->
<!-- RESULTS:training:end -->

![Learning curves](results/grid_b08/learning_curves.png)

### Baseline scenario matrix

```
python scripts/run_benchmark.py
```

<!-- RESULTS:benchmark:start -->
<!-- RESULTS:benchmark:end -->

## Datasets

`dataset_adapters.py` converts the raw files in `data/` into trace CSVs:

| Raw file | What it is | Adapter | Derived trace |
|---|---|---|---|
| `cs448b_ipasn.csv` | Daily flow counts per (local host, remote ASN), 92 days. One simulation step = one day. | `convert_cs448b` | `cs448b_trace.csv` |
| `output1.csv` | Packet capture, numeric export: per-packet timestamp, source IP, destination IP, length. | `convert_packet_capture` | `pcap_trace.csv` |
| `Midterm_53_group.csv` | The same capture exported from Wireshark's text UI. Non-IP rows are dropped. | `convert_packet_capture` | identical to `pcap_trace.csv` |

```
python scripts/prepare_trace.py --in data/cs448b_ipasn.csv --out data/cs448b_trace.csv
python scripts/prepare_trace.py --in data/output1.csv --out data/pcap_trace.csv --bin-seconds 5
python examples/train_agent.py --algo ppo --trace data/pcap_trace.csv --nodes 16 --load 110 --budget 0.5
```

Design points:

* IP addresses and ASNs are bucketed into node ids with a stable MD5-based
  hash, so conversion is reproducible across processes and machines.
* Placement is topology-aware: the busiest buckets go to the best-connected
  nodes, so a traffic hub does not land on a two-link grid corner. For
  `--topology random`, pass `--topology-seed` equal to the env's seed.
* The pcap adapter bins packets into 5-second steps; the cs448b trace uses
  daily bins. The two traces therefore represent very different timescales.
* Trace volume is rescaled so the mean offered load per step, over all time
  bins including empty ones, equals `target_load`.

## Notebooks

All four are committed with executed outputs. To re-run one:

```
cd notebooks
python -m jupyter nbconvert --to notebook --execute --inplace 01_simulation_walkthrough.ipynb
```

| Notebook | What it shows |
|---|---|
| `01_simulation_walkthrough.ipynb` | The five topologies and their nodes, one episode step by step (backlog, utilization, the four reward terms), network renders, `rgb_array` output, a zero-bandwidth stress test and a node bottleneck. |
| `02_policy_comparison.ipynb` | All six baselines on synthetic traffic and on the pcap trace, with 95% intervals. |
| `03_dataset_exploration.ipynb` | The raw datasets, the topology-aware placement, and both traces run through the simulator. |
| `04_ppo_training_curves.ipynb` | Learning curves of the saved multi-seed PPO and SAC runs, plus a short PPO run trained live on the pcap trace. |

## Tests

```
python -m pytest -q
```

<!-- RESULTS:tests:start -->
<!-- RESULTS:tests:end -->

The suite covers Gymnasium API conformance (`check_env`, including rendering),
PettingZoo `parallel_api_test` on all five topologies, packet-unit
conservation, buffer and link-capacity bounds, seeding, input validation, the
reward terms, the baselines, and the dataset pipeline against the real files.

## Limitations

* Time is slotted and packets move one hop per step; there is no propagation
  delay, retransmission or transport-layer behaviour.
* A link's budget is shared by both directions and nodes are served in id
  order, so lower-numbered nodes get first use of a contested link.
* Each node has a single FIFO queue; there is no per-class scheduling inside a
  node. QoS classes differ only in size, deadline and reward weight.
* A packet's route is fixed at admission; routing weights do not re-route
  packets already queued.
* The per-step throughput term is clipped at 1, so a draining backlog and
  steady state look the same to that term (the delay term separates them).
* The multi-agent environment gives every agent the global observation.
* RL results are reported for one training scenario; the scenario matrix
  covers the rule-based baselines only.
* Tested on Python 3.14 (Windows 11). The code parses under the Python 3.10
  grammar but has not been run on 3.10 to 3.13.

## Changes in 0.3.0

Dynamics and reward changed in this release, so results from 0.2.0 are not
comparable.

* `reset()` without a seed no longer replays the same traffic.
* Fractional link service replaces all-or-nothing packet service.
* Node buffers are enforced while traffic is moving.
* Delay and deadline terms now include packets still waiting in queues.
* Added: routing-aware baselines, node bottlenecks, SAC/TD3 training,
  scenario-matrix benchmark, input validation, `rgb_array` rendering.
* Multi-agent env: every agent controls at least one link.

## License

BSD 2-Clause; see `LICENSE`.

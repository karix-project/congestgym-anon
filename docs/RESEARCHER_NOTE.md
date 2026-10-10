# Researcher note

From the Developer role, for the Researcher. It states what the code now is
and what was measured, and passes on open questions that are yours to judge.
I have not edited the slides or the abstract.

Branch `fix/qa-round-1`, version 0.3.0. Evidence files are under `results/`
and `qa/`; details of each fix are in `docs/TESTER_REPORT.md`.

## 1. What the contribution is

A Gymnasium environment (plus a PettingZoo variant) in which one agent sets,
every step, a bandwidth weight and a routing weight for each link of a
store-and-forward packet network, under a total bandwidth budget, with a
four-term QoS reward. Around it: five topologies, synthetic multi-class
traffic, two dataset-driven traces, a node-bottleneck option, six rule-based
baselines (two routing-aware), training scripts for PPO, SAC and TD3, and a
benchmark script that reports confidence intervals.

What it is not: the first Gym-style network environment. NetworkGym and
mobile-env already provide Gym-style network environments, so the novelty
wording on slides 8 and 12 is too strong. A defensible framing is narrower:
*joint* per-link bandwidth allocation and routing-weight control, in a small
packet-level simulator with explicit buffers and QoS deadlines, packaged with
baselines and a reproducible evaluation protocol. Whether that is enough for
the venue you have in mind is your call.

## 2. Model and assumptions

* Slotted time. A packet moves at most one hop per step.
* Store-and-forward with finite node buffers; one FIFO queue per node with
  head-of-line blocking; no per-class scheduling inside a node.
* Links are undirected; both directions share one per-step budget
  (`link_capacity` x bandwidth weight). Nodes are served in id order, so
  lower-numbered nodes get first use of a contested link.
* Fractional service (new in 0.3.0): a packet larger than the remaining link
  budget is transmitted over several steps. Previously it never moved, which
  made every allocation below about 0.4 equivalent.
* A packet enters the next node only if that node had room at the start of the
  step plus arrivals so far; otherwise it is dropped. (New in 0.3.0. Before,
  buffers could hold several times their nominal size; see N1 in the tester
  report. Every result produced with 0.2.0 was affected by this.)
* Routing: Dijkstra over the action's routing weights, computed every step,
  applied to a packet when it is admitted and then fixed.
* Bandwidth budget: the sum of bandwidth weights is capped at `b x E` by
  proportional scaling.
* Traffic: Poisson arrivals in four QoS classes (voice, gaming, video,
  best-effort) with class-specific size, delay sensitivity and deadline;
  about 104 units per step by default against 12 links of capacity 30.
* Observation: per-link utilization and pending demand, per-node buffer
  occupancy, all in [0, 1]. No per-class or per-destination information.

## 3. Reward

`R = T - k*D - c*L - m*M`, defaults k=0.5, c=1.0, m=0.5; range [-2, 1].

* T: delivered / offered units in the step, clipped at 1.
* D: sensitivity-weighted mean age of traffic in the system (delay of packets
  delivered this step and current age of packets still queued), divided by a
  10-step reference, clipped at 1.
* L: dropped / offered units, clipped at 1.
* M: packets that missed their deadline this step (late delivery or expiry in
  a queue), over packets resolved this step.

Changes I made (M1): D and M used to be measured on delivered packets only,
so a completely stalled network reported zero delay and zero deadline misses.
Queued packets now count.

Points for your judgement:

* The terms are per-step ratios combined linearly with hand-chosen weights.
  NetworkGym uses a log-ratio style reward; a comparison or a justification
  of the linear form would strengthen the paper. No sensitivity analysis over
  k, c, m was run.
* T is clipped at 1, so the reward does not credit clearing a backlog beyond
  steady state. D is what distinguishes the two.
* D mixes delivered and queued traffic in one average and is weighted by
  class sensitivity, so it should not be described as "latency".
* M's denominator is the number of packets resolved in the step, which makes
  it noisy when few packets resolve.

## 4. Evaluation protocol

* Episodes of 200 steps; reward averaged over steps 20 to 199 (warm-up
  excluded because the network starts empty).
* Training envs: seeds s .. s+n_envs-1, fresh traffic every episode.
* Learning curves: deterministic policy on validation seeds 1000+.
* Final table: test seeds 2000 to 2009, unseen by training and by the curves.
* RL agents: five training seeds; the 95% Student-t interval is over
  training seeds (each seed's mean over the ten test episodes).
* Baselines: interval over the ten test episodes.
* No hyperparameter search was done for PPO or SAC. PPO: n_steps 512 x 4
  envs, batch 256, gamma 0.95, lr 3e-4, `log_std_init=-1`, reward
  normalisation. SAC: SB3 defaults except gamma 0.95, buffer 200k.
  The baselines have no tuned parameters either.
* Agents act on a [-1, 1] action space mapped linearly to [0, 1].

## 5. Results

Scenario: 3x3 grid, synthetic traffic, load x1, budget 0.8.

| Policy | Kind | Reward (mean ± 95% CI) | n | CI over |
|---|---|---:|---:|---|
| PPO | RL agent | +0.494 ± 0.013 | 5 | training seeds |
| queue_proportional | baseline | +0.404 ± 0.024 | 10 | test episodes |
| congestion_aware | baseline | +0.386 ± 0.021 | 10 | test episodes |
| SAC | RL agent | +0.383 ± 0.028 | 5 | training seeds |
| load_aware_routing | baseline | +0.378 ± 0.026 | 10 | test episodes |
| max_bandwidth | baseline | +0.317 ± 0.030 | 10 | test episodes |
| random | baseline | -0.901 ± 0.026 | 10 | test episodes |
| min_bandwidth | baseline | -1.803 ± 0.008 | 10 | test episodes |

* **PPO** (500,000 steps): held-out reward rose from -0.487 at the first checkpoint
  (25,000 steps) to +0.460 at the last, averaged over seeds. On the test seeds
  it scores +0.494 +/- 0.013, above the best baseline (`queue_proportional`,
  +0.404 +/- 0.024); the intervals do not overlap.
* **SAC** (100,000 steps): held-out reward rose from -0.834 at 10,000 steps to
  +0.347 at 100,000. On the test seeds it scores +0.383 +/- 0.028 (per seed:
  +0.416, +0.395, +0.360, +0.365, +0.381). That is above `max_bandwidth` and far above `random`, and
  statistically indistinguishable from the three stronger heuristics. Its curve
  flattened after about 60,000 steps (+0.329 at 60,000, +0.347 at 100,000), so
  with these untuned settings SAC learns much faster per step than PPO but
  levels off lower. SAC had one fifth of PPO's environment steps; whether more
  steps or tuning would close the gap was not tested.
* Learning curves: `results/grid_b08/learning_curves.png`.

Before the fixes, the same PPO script produced a held-out reward of
-1.000 +/- 0.000 (identical to allocating nothing): the policy's deterministic
action sat in a region where no packet could move.

Baseline scenario matrix (35 scenarios, 10 seeds each,
`results/benchmark/summary.md`): no single heuristic dominates.
`max_bandwidth` is best in 16 scenarios (mostly light load, generous budget,
or the dense mesh), `queue_proportional` in 14 (tight budgets, traces),
`congestion_aware` in 5 (including the node-bottleneck scenario: +0.003
against -0.196 for equal split). On the star topology the routing-aware
baselines equal their bandwidth-only counterparts, as they must: a tree has
no alternative routes.

## 6. What the results do and do not support

Supported:

* An RL agent can be trained in this environment and, in the one scenario
  tested, PPO is clearly better than every rule-based baseline on unseen
  traffic. The claim "learning agents benchmarked against rule-based
  baselines" is supported for that scenario.
* The environment discriminates between policies: the baseline ranking
  changes with topology, load and budget.

Not supported, or not tested:

* Generalisation of the RL result to other topologies, loads, budgets or the
  traces. RL was trained in one scenario only. Notebook 04 contains one
  short single-seed PPO run on the pcap trace; it is a demonstration.
* Any claim about absolute performance figures from the slides (see 7).
* Comparison against established traffic-engineering methods (OSPF/ECMP
  implementations, MPLS-TE, or published RL-TE systems). The baselines here
  are simple heuristics written for this simulator.
* Scalability: the largest network run is 16 nodes.
* Multi-agent learning: the environment exists and passes API tests; nothing
  was trained on it.
* Statistical testing beyond 95% intervals (no hypothesis tests, five seeds).
* DQN, which the slides name: the action space is continuous, so PPO and SAC
  were used. DQN would need a discretised action space.

## 7. Open items passed on from QA (yours to decide)

* **Name clash** with the CARLA autonomous-driving simulator.
* **Dataset provenance, licence and privacy.** The public repository holds
  about 74 MB of raw data, including a Wireshark capture with real IP
  addresses (`output1.csv`, `Midterm_53_group.csv`) and a Stanford CS448B file
  with no stated data licence. `LICENSE` is BSD-2-style and carries a personal
  email address. I did not remove or alter any raw dataset; the user has been
  asked to decide.
* **Timescale mismatch.** The cs448b trace uses 92 daily bins as simulation
  steps; the pcap trace uses 5-second bins. In both cases only the relative
  shape of the load survives, because volume is rescaled to a target load.
  The cs448b file contains flow counts, not volumes.
* **Unverified claims in the presentation.** "Significantly lower queuing
  delays, minimal packet loss" (slide 6) and the 39.13% / 44.57% / 99.42%
  figures (slides 6 and 11) come from other papers, not from this project.
  This project's own measured figures are in section 5.
* **Citation and document errors.** NetworkGym is listed as "M. Yin et al."
  but the first author is Momin Haider; mobile-env is cited as IEEE Network
  but is a NOMS 2022 paper; Yu, Wang and Liew is cited as "IEEE Transactions"
  while the reference list gives arXiv:1712.00162; the slides say SB3
  "requires PyTorch 2.8" (unverified; this project ran SB3 2.9.0 on PyTorch
  2.14.1); roll number B22IT072 against B23 for the others; "S. Sathish"
  (slides) against "Sathish Kumar" (abstract). I have not checked these
  against the sources; they are relayed from the QA report.
* **Framing of "learning agents benchmarked".** See section 6.

## 8. Limitations I know of

Listed in the README. The ones most relevant to a reviewer: slotted one-hop
time with no propagation delay or transport behaviour; shared undirected link
budget with id-ordered service; single FIFO per node; routes fixed at
admission; small networks; one RL training scenario; untuned
hyperparameters; tested on a single Python version and OS.

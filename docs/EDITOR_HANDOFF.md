# Handoff to the Editor/Presenter

Written by the Developer role, 11 Oct 2026. Purpose: everything needed to
draft the project report and the PowerPoint for the university presentation,
without re-reading the whole history.

**Project:** Carla-Gym: A Gymnasium-Based Reinforcement Learning Framework for
Congested Network Traffic Scenarios. KITS Warangal, Dept. of IT, Batch 14
(K. Aryan, K. Sai Akshay, D. Karthik Kumar, T. Varshitha). Supervisor
Sri G. Yeshwanth. Major Project, VII semester.
**Code:** https://github.com/spaciouskarter78/carla-gym, branch `main`,
version 0.3.0.

## 0. Before you start (project rules)

* This role runs on **Sonnet**, and only starts once the user has supplied a
  **template or a sample project report**. Ask for it first.
* The deliverable is two documents: a general project report and a research
  paper, plus the presentation.
* Write in a natural, human voice. Ask which role the user wants at the end of
  every reply, and warn before any step that could hit a token limit.
* **Do not invent numbers.** Every figure below was produced by a run and is
  stored in the repository at the path given. If you need a number that is
  not here, ask for the Developer role to produce it.

## 1. What the project is, in plain words

Carla-Gym is a Python library that turns a congested computer network into a
reinforcement-learning problem using the standard Gymnasium interface. At
every time step an agent decides two things for every link in the network:
how much of its bandwidth to switch on, and how attractive the link should be
for routing. The agent is rewarded for delivering traffic and penalised for
delay, packet loss and missed deadlines. The library includes several network
shapes, synthetic and trace-driven traffic, a way to degrade a node to create
a bottleneck, six rule-based policies to compare against, and scripts that
train RL agents and benchmark them.

One sentence for a slide: *a lightweight, laptop-friendly test bed where an
RL agent learns to allocate bandwidth and steer routes in a congested
network, and can be compared fairly against rule-based traffic engineering.*

## 2. What was built (for a "system design" or "implementation" section)

| Part | What it does | File |
|---|---|---|
| Environment | Gymnasium `Env`: observation, action, reward, episode handling | `carla_gym/env.py` |
| Network simulator | Store-and-forward packets, finite node buffers, link capacity, one hop per step, FIFO queues | `carla_gym/core/network.py` |
| Traffic | Poisson arrivals in four QoS classes (voice, gaming, video, best-effort) | `carla_gym/core/traffic.py` |
| Trace traffic | Replays a load pattern derived from real measurements | `carla_gym/core/trace_traffic.py` |
| Reward | `R = Throughput - k*Delay - c*PacketLoss - m*DeadlineMiss` | `carla_gym/utils/reward.py` |
| Baselines | Six rule-based policies, two of them routing-aware | `carla_gym/baselines.py` |
| Multi-agent version | Same simulator through the PettingZoo API | `carla_gym/marl_env.py` |
| Training | PPO, SAC, TD3 with held-out evaluation | `examples/train_agent.py` |
| Benchmark | Baselines over 37 scenarios with confidence intervals | `scripts/run_benchmark.py` |
| Notebooks | Four executed notebooks with plots | `notebooks/` |

Key design facts:

* **Topologies:** grid, ring, star, mesh, random. Default is a 3x3 grid: 9
  nodes, 12 links.
* **Observation:** link utilisation, pending demand per link, node buffer
  occupancy; size `2E + N` (33 for the default grid).
* **Action:** one bandwidth weight and one routing weight per link; size `2E`
  (24 for the default grid). A bandwidth budget (default 0.8) caps the total.
* **Default reward weights:** k = 0.5, c = 1.0, m = 0.5; reward lies in [-2, 1].
* **Software actually used:** Python, Gymnasium, Stable-Baselines3, PyTorch,
  NetworkX, NumPy, Pandas, Matplotlib, PettingZoo, pytest, Jupyter.
* **Hardware actually used:** a laptop CPU. No GPU. One PPO training run of
  500,000 steps takes about ten minutes on its own.

## 3. Results you may quote

### 3.1 Headline: RL agent against rule-based policies

Scenario: 3x3 grid, synthetic traffic, bandwidth budget 0.8, 200-step
episodes. Five independent training runs per algorithm. Scores are on ten
test traffic sequences the agents never saw in training.
Source: `results/grid_b08/summary.md` and `results/grid_b08/qos.md`.

| Policy | Reward (mean ± 95% CI) | Traffic delivered | Traffic dropped |
|---|---:|---:|---:|
| **PPO** (RL, 500k steps) | **+0.494 ± 0.013** | **92.2%** | **7.7%** |
| queue_proportional (best rule-based) | +0.404 ± 0.024 | 87.0% | 13.0% |
| congestion_aware (routing-aware) | +0.386 ± 0.021 | 86.6% | 13.4% |
| SAC (RL, 100k steps) | +0.383 ± 0.028 | 87.6% | 12.3% |
| load_aware_routing (routing-aware) | +0.378 ± 0.026 | 86.2% | 13.9% |
| max_bandwidth (equal split) | +0.317 ± 0.030 | 82.6% | 17.3% |
| random | -0.901 ± 0.026 | 39.4% | 60.3% |

(The delivered and dropped columns come from a separate evaluation pass; its
reward for `random` is -0.898 because that policy draws fresh random actions.
All other rows are identical in both passes.)

Safe statements:

* PPO beats every rule-based policy. Against the best one the gain is
  +0.090 ± 0.013, and PPO is better in all 50 paired comparisons (5 training
  runs x 10 test sequences). Source: `qa/research/analysis.log`.
* In network terms PPO delivers about 5 percentage points more traffic and
  drops about 40% fewer packets than the best rule-based policy (7.7% against
  13.0%).
* **Delay did not improve**: 3.89 steps for PPO against 3.88. Do not claim
  lower delay.
* SAC learned to the level of the better rule-based policies but did not beat
  them. Do not say "PPO is better than SAC": SAC was trained for a fifth of
  the steps.
* The learned policy is adaptive: if PPO's own average action is simply held
  constant, its reward falls from +0.498 to +0.303.
* Both controls matter: forcing plain shortest-path routing costs the PPO
  agent 0.052 ± 0.017 reward. Source: `qa/research/routing_ablation.log`.

### 3.2 Before and after the fixes (a good "challenges" slide)

The first working version could not train an agent: PPO scored exactly -1.000,
the same as allocating no bandwidth at all. Four faults were found and fixed:

1. Every episode replayed identical traffic.
2. A packet larger than a link's per-step budget never moved, so most low
   allocations were equally useless.
3. Node buffers silently held up to eight times their stated size.
4. Packets stuck in queues were invisible to the delay and deadline penalties.

After the fixes PPO went from -1.000 to +0.494.

### 3.3 Benchmark across scenarios (rule-based policies only)

37 scenarios x 6 policies x 10 seeds = 2,220 episodes. Source:
`results/benchmark/summary.md`, also in the README.

* No single rule-based policy wins everywhere; which is best depends on
  topology, load and budget.
* Only 10 of the 37 scenarios have a clear winner; in the other 27 the
  leading policies are statistically tied. Say "tied" where they are tied.
* In the node-bottleneck scenario the routing-aware policy `congestion_aware`
  is the best (+0.003 against -0.196 for equal split).
* Trace-driven scenario (packet-capture pattern, mean load 40, budget 0.5):
  `queue_proportional` +0.240 with 77% delivered, against -0.002 and 59% for
  equal split.

### 3.4 Quality and testing

* 129 automated tests pass (115 run in a fresh copy of the repository; 14
  need raw data that is not distributed).
* The environment passes Gymnasium's own conformance checker and PettingZoo's
  parallel API test on all five topologies.
* Saved models reproduce their reported scores exactly.
* The work went through independent testing and research review rounds; the
  records are `docs/TESTER_REPORT.md`, `docs/QA_RETEST_REPORT.md` and
  `docs/RESEARCHER_REVIEW.md`.

## 4. Figures you can use

| Figure | Where |
|---|---|
| Learning curves, PPO and SAC against the baselines | `results/grid_b08/learning_curves.png` |
| The five topologies; network state renders; reward terms over time; a node bottleneck | `notebooks/01_simulation_walkthrough.ipynb` |
| Bar charts of the six baselines, synthetic and trace | `notebooks/02_policy_comparison.ipynb` |
| Load over time and traffic matrices of the two traces; placement on the grid | `notebooks/03_dataset_exploration.ipynb` |
| Learning curves with per-seed lines and confidence band | `notebooks/04_ppo_training_curves.ipynb` |

Open a notebook and export the images, or ask the Developer role to save
them as PNG files. The logo is `.github/carla-logo.png`.

## 5. Things you must NOT write

These came out of the research review. Each would be challenged by an examiner.

| Do not write | Why | Write instead |
|---|---|---|
| "Novel", "first", "no standardized RL benchmarks for networking exist" | Several exist: NetworkGym, mobile-env, ns3-gym, PRISMA, PackeRL, Park. The slides themselves cite two. | "A lightweight environment for *joint* bandwidth and routing control, which the existing environments we found do not offer in this form." |
| ns-3 is a "legacy" simulator without Gymnasium support | ns-3 is the current standard and ns3-gym connects it to Gym | "ns-3 based environments are high-fidelity but heavy to set up and slow to train on." |
| "Significantly lower queuing delays, minimal packet loss" | Delay is unchanged; loss is 7.7% against 13.0% | Quote the measured figures and name the scenario |
| "PPO and DQN" | DQN was not used | "PPO and SAC" |
| RL was evaluated "across diverse scenarios" | RL agents were trained in **one** scenario. The 37 scenarios are for rule-based policies. | Say exactly that |
| Results on "real traffic" | The traces are load *shapes* derived from real measurements and rescaled | "trace-driven load patterns" |
| Anything implying real-network accuracy | The simulator was not validated against ns-3 or a real network | "an abstract packet-level model" |
| Multi-access steering across Wi-Fi, LTE and 5G as an application | Not modelled by this project | Drop it, or mark it as future work |
| GPU required | Everything ran on a laptop CPU | "No GPU required" |
| 39.13%, 44.57%, 99.42% as if they were this project's results | They are from other papers | Attribute them, and see section 6 |

## 6. Corrections needed in the existing slides and abstract

The full slide-by-slide table is `docs/SLIDES_ABSTRACT_REVIEW.md`, section 2.
The abstract is accurate and needs no factual change. The important slide
fixes:

* **Slide 3, 8, 10, 12:** remove the "novel" and "no benchmarks exist" claims
  (section 5 above).
* **Slide 4 (diagram):** "PPO, DQN" becomes "PPO, SAC"; the reward formula
  needs the fourth term, `- m*DeadlineMiss`.
* **Slide 5:** NetworkGym is by **M. Haider et al.**, not M. Yin. mobile-env
  is **NOMS 2022**, not IEEE Network (slide 16 already has it right). "Proves"
  should be "reports".
* **Slide 6:** replace the "observed outcomes" line with measured results;
  remove the Wi-Fi/LTE/5G application.
* **Slide 7:** the two figures appear to come from the mobile-env and
  NetworkGym papers. Add a credit line under each.
* **Slide 11:** the 39.13% delay reduction in Habib et al. is against a
  threshold heuristic, not "flat DRL" (against DQN it is 27.74%). The 44.57%
  and 99.42% figures could not be confirmed from abstracts; check them in the
  full papers or drop them. Add a reference column.
* **Slide 14:** remove Ray RLlib, SciPy and TensorBoard (not used). "PyTorch
  2.8 or newer" is correct.
* **Slide 15:** no GPU is needed.
* **Names:** the coordinator is "Sri. S. Sathish" on the slides and
  "Sri. S. Sathish Kumar" on the abstract; "Chairman" against "Head of the
  Department". Use one form. Roll number B22IT072 appears in both documents;
  only the team can confirm it.

## 7. Honest limitations (for a "limitations and future work" section)

* RL agents were trained in a single scenario on a 9-node network.
* The simulator is an abstraction: one hop per step, shared link capacity for
  both directions, no backpressure. It has not been validated against ns-3.
* The reward averages per-step ratios, so on very bursty traffic it does not
  track delivered traffic. Always show delivered and dropped percentages too.
* Hyperparameters were not tuned.
* Tested on Python 3.14 on Windows 11 only.
* The multi-agent environment works but no multi-agent training was done.

Future work already scoped (see `docs/PUBLICATION_READINESS.md`): RL training
on more scenarios, agents trained with one control switched off, ECMP and
optimisation-bound baselines, larger networks, validation against ns-3.

## 8. Suggested outline

**Report:** Introduction; Literature review (with corrected citations);
Problem statement and objectives; System design (section 2); Implementation
and testing (section 3.4, section 3.2); Results (3.1, 3.3 with figures);
Limitations and future work (section 7); Conclusion; References.

**Presentation (about 15 slides):** title; motivation; related work and the
gap, stated modestly; objectives; architecture diagram (corrected); how the
simulator works; reward; baselines; the "it could not learn, here is why"
story (3.2); headline result table and learning curve (3.1); what the agent
learned (adaptive, both controls matter); benchmark across scenarios (3.3);
testing and reproducibility; limitations and future work; conclusion.

## 9. Things to tell the audience if asked

* **Was AI used?** Yes. An AI assistant was used heavily for coding, testing
  and review. Follow the university's rule on declaring it; the team should
  be ready to explain any part of the code.
* **Where is the data?** Only two derived load traces are distributed. The
  raw files were removed from the current repository because one contains
  real IP addresses and the other has no stated licence. They are still in
  the repository's older history, by the team's decision.
* **Why "Carla"?** Be ready for the question: the name is shared with the
  CARLA driving simulator. No decision has been made to rename.

# Researcher review

Researcher role, 10 Oct 2026 (Opus 5.5). Reviewed: `docs/RESEARCHER_NOTE.md`,
`docs/TESTER_REPORT.md`, `docs/QA_RETEST_REPORT.md`, the result files under
`results/`, and the code where a claim depended on it. Branch
`fix/qa-round-1` at `8de3f9a`. No code was changed in this pass.

New analyses run for this review (they read saved results and saved models,
and train nothing): `qa/research/analysis.py`, `qa/research/routing_ablation.py`,
with logs next to them.

I did not have the slides or the abstract document. Statements about them
below rest on what the QA report relayed.

## 1. Verdict

**The work is scientifically sound for what it now claims, and the central
result is real.** In the tested scenario a PPO agent learns a closed-loop
policy that beats every rule-based baseline on unseen traffic, by a margin
that survives a paired test, and the gain comes from both halves of the
action (bandwidth and routing).

**It is ready to submit as the coding part of the project.** It is **not yet
ready for a peer-reviewed journal**, and for a peer-reviewed conference it
needs the additions in section 5 first. The gap is breadth of evidence and
positioning against prior work, not correctness.

Three inconsistencies need the Developer and then the Tester (section 4).
Several presentation claims must change (section 6); those belong to the
Editor/Presenter later.

## 2. What the evidence supports

### 2.1 PPO vs the baselines (paired, same test seeds)

The reported table compares intervals built in two different ways (training
seeds for agents, test episodes for baselines). A paired comparison is the
cleaner test: for each training seed, the agent's reward minus the
baseline's on the same ten test episodes.

| Comparison | Mean difference (95% CI over 5 training seeds) | Agent better in |
|---|---|---|
| PPO - `queue_proportional` | **+0.090 +/- 0.013** | 50 of 50 (seed, episode) pairs |
| PPO - `congestion_aware` | +0.108 +/- 0.013 | 50 of 50 |
| PPO - `max_bandwidth` | +0.177 +/- 0.013 | 50 of 50 |
| SAC - `queue_proportional` | -0.021 +/- 0.028 | 13 of 50 |
| SAC - `congestion_aware` | -0.003 +/- 0.028 | 26 of 50 |
| SAC - `max_bandwidth` | +0.066 +/- 0.028 | 49 of 50 |

So: PPO is better than the best heuristic, without exception across seeds
and episodes. SAC is better than equal split and no better than the two
queue-aware heuristics. Both statements match what the Developer reported.

Scale of the effect: the PPO gain over the best heuristic is 0.090 reward,
about 7% of the distance between a random policy and that heuristic. In
network terms (aggregate over the test episodes, PPO seed 0): delivered
traffic 92.1% against 86.9%, dropped traffic 7.8% against 13.0%. The delay
and deadline terms are unchanged. The agent wins by losing fewer packets,
not by being faster.

### 2.2 The policy is adaptive, not a lucky static allocation

A reviewer will ask whether the agent just found a good fixed setting. It did
not:

| | PPO (mean of 5) | SAC (mean of 5) |
|---|---|---|
| Learned policy, reacting to observations | +0.498 | +0.381 |
| Its own average action, held constant | +0.303 | +0.260 |
| Its own actions replayed in random order | +0.336 | +0.256 |

Removing the feedback costs PPO 0.16 to 0.20 reward and drops it below the
equal-split baseline (+0.317). The policy's value is in *when* it does what.

### 2.3 Both halves of the action matter

Test-time ablation on the five PPO agents:

| Variant | Reward (mean +/- 95% CI) |
|---|---|
| Full agent | +0.494 +/- 0.013 |
| Agent's bandwidth, routing forced to uniform (hop-count shortest path) | +0.442 +/- 0.016 |
| Agent's routing, bandwidth from `queue_proportional` | +0.456 +/- 0.008 |
| `queue_proportional` alone | +0.404 |

Learned routing is worth +0.052 +/- 0.017. The agent's routing also lifts the
heuristic's bandwidth rule from +0.404 to +0.456, and its bandwidth alone
still beats the heuristic. About 15% of source-destination routes change
from one step to the next under the agent. This is direct support for the
abstract's "dynamic packet routing" and "bandwidth allocation" as jointly
learned controls, and it is the most publishable single finding here. It is
an ablation at test time, not agents trained with one half disabled; say so.

### 2.4 The baseline matrix

35 scenarios, 6 policies, 10 seeds. The statement "no single heuristic
dominates" is true but needs tempering:

* the best heuristic is separated from the runner-up by non-overlapping
  intervals in only **9 of 35** scenarios;
* in **8** scenarios the four real heuristics are within 0.03 of each other
  (the environment does not discriminate between them there);
* in **6** scenarios every heuristic scores below -1.0: the network is so
  overloaded that no policy can do well, and those rows say little.

The matrix shows the environment produces a range of regimes. It does not
show 35 informative comparisons.

## 3. Soundness of the model and the method

Sound, with stated limits:

* **Simulator.** A slotted store-and-forward model with finite buffers is a
  legitimate abstraction. After the QA rounds it conserves packets, respects
  buffer and link capacity, and has no dead zone. It has **not been validated
  against a packet-level simulator** (ns-3, OMNeT++) or a testbed. Any
  sentence implying real-network fidelity is unsupported; "abstract
  packet-level model" is accurate.
* **Reward.** Bounded, each term interpretable. The known quirks (T5, T7 in
  the QA report) are small and documented. The linear weights are arbitrary
  and untested; that is normal for a first paper but must be stated.
* **Evaluation protocol.** Train / validation / test seed separation is
  correct, and I found no leakage. Five training seeds is the usual minimum.
  Reproducibility is good: saved models reproduce exactly, a full re-run of
  one PPO seed lands at +0.492 against +0.496.
* **Fairness of the comparison.** Neither the agents nor the heuristics were
  tuned, which is even-handed. PPO had 500k steps and SAC 100k, so "PPO beats
  SAC" is **not** a supported claim; "SAC learned faster per step and
  levelled off lower under these settings" is.
* **A metric problem (see 4.1).** The reported `throughput` is the mean of
  per-step clipped ratios (0.844 for PPO), not the fraction of traffic
  delivered (0.921). The two differ by eight points.

## 4. Inconsistencies: for the Developer, then the Tester

1. **Reported throughput and loss are not the quantities a reader will
   assume.** Tables, notebooks and the benchmark print the mean of per-step
   ratios. Episode-aggregate delivered/offered and dropped/offered should be
   reported next to them (the raw counters are already in `info`). This
   changes no reward and no training result. Same for latency: report the
   delivered-packet mean and a tail percentile per QoS class if it is cheap.
2. **`results/benchmark/summary.md` and the README bold the best policy in
   every row**, including the 26 rows where the best two are statistically
   tied. Bold only when the intervals do not overlap, or mark ties.
3. **`docs/RESEARCHER_NOTE.md` says NetworkGym "uses a log-ratio style
   reward".** I could not confirm this and it should be removed or sourced
   before anyone repeats it.

None of these affects the correctness of the code or the trained results.

## 5. What a peer-reviewed venue would require

In rough order of value for effort:

1. **RL in more than one scenario.** At minimum: a second topology, the
   node-bottleneck scenario, and one trace, each with 5 seeds. This is the
   biggest gap; today the RL claim rests on a single 9-node grid.
2. **Trained ablations**: bandwidth-only and routing-only agents, to back the
   test-time ablation in 2.3.
3. **Equal budgets or a convergence argument** for PPO vs SAC.
4. **A recognised traffic-engineering baseline** (ECMP, or OSPF with
   inverse-capacity weights, or a max-min fair allocation) with a citation,
   so the comparison is not only against heuristics written for this
   simulator.
5. **Scale**: at least one network of 30 to 50 nodes, with run time reported.
6. **Reward-weight sensitivity** (k, c, m) on the baseline ranking.
7. **Validation of the simulator** against ns-3 on one small case, or an
   explicit statement that this is future work.
8. **Related work done properly**: NetworkGym, mobile-env, ns3-gym, and the
   RL traffic-engineering literature (for example DRL-TE, and routing-with-RL
   work such as Valadarsky et al.), with a table of what each environment
   offers. I name these from memory; check each before citing.
9. **Dataset provenance** settled (section 7).

Items 1 to 3 need only compute and the existing scripts. For a student
conference or a workshop, items 1, 4 and 8 would probably be enough.

## 6. Claims that must change (for the Editor/Presenter)

| Claim (as relayed by QA) | Status |
|---|---|
| Novelty wording on slides 8 and 12 | Too strong. Gym-style network environments exist. Defensible: joint per-link bandwidth and routing-weight control in a lightweight packet-level simulator, with baselines and a reproducible protocol. |
| "Significantly lower queuing delays, minimal packet loss" (slide 6) | Not supported by this project. Measured: delay unchanged; loss 7.8% for PPO against 13.0% for the best heuristic, in one scenario. |
| 39.13% / 44.57% / 99.42% (slides 6, 11) | From other papers. Must be attributed to them or removed. |
| "PPO and DQN" | DQN was not used and cannot be used directly on a continuous action space. Say PPO and SAC. |
| "Benchmarked across diverse scenarios" | True for the baselines (35 scenarios). RL agents: one scenario. |
| SB3 "requires PyTorch 2.8" | Not checked against the SB3 documentation. This project ran SB3 2.9.0 on PyTorch 2.14.1. Remove the version claim or cite the SB3 docs. |

### Citations (checked against the sources this session)

* **NetworkGym**: authors are Momin Haider, Ming Yin, Menglei Zhang, Arpit
  Gupta, Jing Zhu, Yu-Xiang Wang; NeurIPS 2024 (Datasets and Benchmarks
  track). Cite as "M. Haider et al.", not "M. Yin et al.". Confirmed.
* **mobile-env**: S. Schneider, S. Werner, R. Khalili, A. Hecker, H. Karl,
  IEEE/IFIP NOMS 2022. Not IEEE Network. Confirmed.
* **Yu, Wang, Liew**, "Deep-Reinforcement Learning Multiple Access for
  Heterogeneous Wireless Networks": arXiv:1712.00162; the search confirmed a
  conference version at IEEE ICC 2018. I believe a journal version appeared
  in IEEE JSAC in 2019, but I could not confirm that this session; check IEEE
  Xplore. "IEEE Transactions" is wrong either way. Note also that this paper
  is about MAC-layer multiple access, not routing or bandwidth allocation, so
  check that it is cited for the right point.
* Roll number (B22IT072 vs B23) and the supervisor-name variants: I cannot
  check these; the team must.

## 7. Open items that are the user's decision

* **Raw datasets in the public repository.** `output1.csv` and
  `Midterm_53_group.csv` are a packet capture with real IP addresses. I
  searched for the origin and licence of `cs448b_ipasn.csv` and found neither.
  Publishing data of unknown licence, or addresses that may be
  identifiable, is the one real liability left. My recommendation: keep only
  the derived traces (they contain node ids 0 to 15, no addresses), state the
  provenance of each raw file in the README, and remove the raw files from
  the repository and its history unless a licence is found. That needs a
  Developer change (tests and notebook 03 read the raw files).
* **Timescales.** The cs448b trace treats one day as one step and contains
  flow counts, not volumes; the pcap trace uses 5-second bins. After
  rescaling only the shape of the load survives. Describe both as "load
  shapes derived from real traces", not as real traffic replay.
* **Name.** "CarlaGym" will be confused with the CARLA driving simulator, and
  searches for it return driving-simulator projects. For a publication I
  would rename; for the college submission it is tolerable.
* **Personal email in `LICENSE`.** No scientific bearing.

> Update, 10 Oct 2026 (after this was written): the user decided to remove the raw datasets.
> They are no longer in the repository's current files; only the derived traces remain.
> They are still in the git history. See `docs/TESTER_REPORT.md`, section 13.

## 8. Summary for the team

* Correct to claim: a Gymnasium environment for joint bandwidth and routing
  control; PPO learns an adaptive policy that beats six rule-based baselines
  in the tested scenario (+0.090 +/- 0.013 over the best, 50 of 50 paired
  wins); both action halves contribute; SAC reaches heuristic level.
* Do not claim: novelty as the first such environment, lower delay, results
  across scenarios for RL, DQN, real-network fidelity, or any percentage from
  other papers.
* Next: Developer for section 4 (small), Tester to confirm, then the user's
  dataset decision. The section 5 experiments are for a paper, not for the
  14 October submission.

---

# Final Researcher pass (11 Oct 2026, on `main` at `e71c43c`)

Done after rounds 2 to 6 and the merge. New analysis:
`qa/research/trace_load_sweep.py` and its log; the two earlier analyses were
re-run on the merged code (`*_final.log`).

## F1. Verdict

**Unchanged, and now firmer. The coding part is scientifically sound for what
it claims and is fit to submit.** It is still not journal-ready; the list in
section 5 stands. There is one new scientific finding (F4) that should be
documented before anyone publishes results on the cs448b trace. It does not
affect the headline result.

## F2. The three inconsistencies from section 4

| # | Item | Status on `main` |
|---|---|---|
| 1 | Throughput / loss reporting | Resolved. Episode-aggregate tables for agents and for all 35 baseline scenarios are in the README, with an explanation of how they differ from the per-step reward terms. |
| 2 | Bolding ties | Resolved. 9 rows bold, 26 marked as ties, with a legend. |
| 3 | Unsourced NetworkGym statement | Resolved. Removed. |

## F3. The earlier evidence still holds

Both analyses give byte-identical output on the merged code: PPO minus the
best heuristic +0.090 +/- 0.013 (50 of 50 paired wins); the policy loses 0.16
to 0.20 without feedback; learned routing is worth +0.052 +/- 0.017. The later
rounds changed validation, reporting and data packaging, not the simulator,
so this is expected, and it is now checked.

The aggregate table adds the plain-language version of the headline: PPO
delivers 92.2 +/- 0.5 % of offered traffic and drops 7.7 +/- 0.5 %, against
87.0 % and 13.0 % for the best heuristic, with the same delay (3.89 against
3.88 steps). **For the synthetic scenario the reward and the aggregate
figures agree on the ranking of every policy**, so the reward is a faithful
objective there.

The README makes no claim of novelty, real-network fidelity or statistical
significance beyond its tables, and states that RL results cover one
scenario. I found nothing in it to retract.

## F4. New finding: the trace scenarios, and a limit of the reward

Round 4 showed that both trace scenarios drop more than half their traffic.
I swept the target load to see whether that is a calibration choice or a
property of the data (grid-16, budget 0.5, 5 seeds):

**pcap trace**

| Target load | `max_bandwidth` reward, delivered % | `queue_proportional` reward, delivered % | qp minus max (95% CI) |
|---:|---|---|---|
| 20 | +0.170, 83.4 | +0.232, 92.7 | +0.062 +/- 0.026 |
| 40 | +0.012, 59.8 | +0.246, 77.1 | +0.234 +/- 0.029 |
| 60 | -0.156, 45.1 | +0.139, 61.9 | +0.295 +/- 0.026 |
| 80 | -0.283, 37.1 | +0.046, 52.4 | +0.329 +/- 0.025 |
| 110 | -0.422, 29.0 | -0.037, 43.1 | +0.386 +/- 0.028 |

This trace is a usable scenario. It was simply run too hot: at the load of
110 used throughout the repository it is deep in overload, while at 20 to 40
it sits in a sensible regime (77 to 93 % delivered) and still separates the
heuristics clearly. Reward and delivered fraction move together.
**Recommendation: make about 30 to 40 the documented load for this trace.**

**cs448b trace**

| Target load | `max_bandwidth` reward, delivered % | `queue_proportional` reward, delivered % |
|---:|---|---|
| 20 | +0.213, 52.3 | +0.110, 52.1 |
| 40 | +0.342, 46.8 | +0.295, 47.6 |
| 60 | +0.403, 45.3 | +0.395, 46.6 |
| 80 | +0.357, 43.6 | +0.441, 46.7 |
| 110 | +0.176, 37.4 | +0.414, 43.8 |

This one is not a meaningful benchmark as configured, for two reasons.

1. **About half the traffic is lost at every load, even at 20.** The loss is
   set by the shape of the data, not by the load level: the busiest 10 % of
   time bins carry 64 % of the volume, the peak bin is 37 times the mean, and
   one node pair carries 56 % of everything. A few spike days overwhelm the
   network whatever the average is. (The source is daily flow *counts* used
   as volumes, with one day treated as one step.)
2. **The reward moves the wrong way.** From load 20 to 60 the reward of equal
   split rises from +0.21 to +0.40 while the delivered fraction *falls* from
   52 % to 45 %. The reward averages per-step ratios, so it gives a quiet
   step the same weight as a spike step; when nearly all the volume sits in
   a few steps, the average says almost nothing about what happened to the
   traffic.

Point 2 is a genuine limit of the reward design, not only of this trace:
**under strongly bursty load, maximising the mean per-step reward is not the
same as maximising delivered traffic.** An agent trained on such a trace
could score well while most traffic is lost. It did not bite in the headline
experiment (stationary Poisson load, reward and aggregates agree), but it
must be stated as a limitation, and any RL result on a bursty trace must
report the aggregate table next to the reward. A volume-weighted reward
(weighting each step by its offered load) is the natural fix to study; that
is a research change needing retraining, not a patch.

## F5. For the Developer (small, optional before submission)

1. Document F4: in the README limitations, and next to the trace rows of the
   benchmark. State that the cs448b scenario is included as a stress case, not
   as a benchmark.
2. Change the suggested load for the pcap trace from 110 to about 30 to 40
   in the README, `prepare_trace.py`'s printed suggestion and notebooks, or
   add those loads to the scenario matrix. This changes published trace
   numbers, so it needs the benchmark re-run and the README regenerated.
3. Not required for the 14 October submission: item 1 is a paragraph of
   text; item 2 is about fifteen minutes of compute plus regeneration.

Then the Tester confirms, as usual.

## F6. What I could not assess

* The slides and the abstract (not provided). Section 6 lists what must
  change in them, on the QA report's account of their contents.
* Whether the venue the team has in mind accepts an environment paper with
  one RL scenario. For a college project submission the evidence is ample.
* Python 3.10 to 3.13, Linux, macOS: untested on this machine.

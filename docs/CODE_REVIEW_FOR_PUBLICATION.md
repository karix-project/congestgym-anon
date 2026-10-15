# Code review for publication

Researcher role, 15 Oct 2026 (Opus 5.5). Reviewed: `main` at `7a06bb7`
(134 tests pass). This complements `docs/PUBLICATION_READINESS.md`
(11 Oct), which judged the *evidence*. This document judges the *code*, read
the way a networking reviewer reads a paper's model section. Every number
below comes from `qa/research/probe_model.py` and `qa/research/probe_bias.py`
and their `.log` files (grid 3x3, budget 0.8, load x1, test seeds 2000-2009,
200 steps, 20 warm-up).

## 1. Short answer

The code is correct and well tested; nothing found here is a crash or a
broken invariant. But four properties of the model would be caught by a
reviewer, and two of them weaken the paper's central claim directly.
**Three of them change the dynamics, so fix them before launching the
8-hour training matrix** (`docs/DEVELOPER_RESUME.md` section 5), or the
runs will have to be repeated.

## 2. New findings

### F1. Deterministic tie-breaking makes symmetric nodes unequal (fix before training)
With uniform routing weights, many source-destination pairs on a grid have
several equal-cost paths, and Dijkstra always picks the same one. Traffic
piles onto some links. Nodes in symmetric positions should get the same
service under uniform traffic; they do not:

| Policy | Corners 0 / 2 / 6 / 8, delivered fraction of their traffic | Node 0 minus node 8 |
|---|---|---|
| max_bandwidth | 0.716 / 0.875 / 0.718 / 0.957 | -0.241 |
| queue_proportional | 0.788 / 0.893 / 0.810 / 0.967 | -0.179 |
| congestion_aware | 0.881 / 0.892 / 0.859 / 0.883 | -0.003 |

Separating the causes (max_bandwidth):

| Tie-break | Service order | Delivered | Spread across sources |
|---|---|---|---|
| fixed (shipped) | id order (shipped) | 0.826 | 0.241 |
| fixed | shuffled each step | 0.804 | 0.235 |
| random among equal costs | id order | 0.787 | 0.138 |
| random among equal costs | shuffled each step | 0.800 | 0.095 |

* Tie-breaking causes most of the bias; service order a smaller part.
* The README limitation ("lower-numbered nodes get first use of a contested
  link") is true of the mechanism but **the measured outcome favours the
  highest-numbered corner**. The text must be corrected.
* Random tie-breaking *lowers* total delivery here, so the routing gain of
  learned and load-aware policies is not just a repair of this artefact.
  It is a fairness problem, and a reviewer will see it as one.

Fix: break ties at random (seeded from the env RNG) and serve nodes in a
random or round-robin order each step; report a per-source fairness figure
(Jain's index of per-source delivered fraction) next to delivered and
dropped. Effort: half a day plus tests.

### F2. Head-of-line blocking wastes a sixth to a fifth of the allocated bandwidth (fix before training)
A node holds one FIFO for all outgoing links. When the head packet cannot
finish on its link, the node stops, including packets bound for *other*
links that still have budget. Share of allocated link budget left unused in
a step while packets that needed that link sat blocked:

| Policy | Unused allocated budget with traffic waiting for it |
|---|---|
| max_bandwidth | 16.1% |
| queue_proportional | 18.6% |
| congestion_aware | 19.7% |

The paper's headline action is per-link bandwidth allocation, and the
simulator throws away a fifth of what is allocated. A reviewer can argue the
learned gain partly comes from working around a simulator artefact rather
than from allocation. Real routers queue per output port.

Fix: add `queueing="per_link"` (one FIFO per outgoing link, sharing the
node buffer) and make it the default for the paper; keep the shared FIFO as
an option and a documented stress setting. Effort: 1 to 2 days, mostly tests
of conservation and buffer bounds.

### F3. The random topology changes size with the seed (fix before training)
`NetworkModel` uses the env seed for both the graph and the traffic. On
9 nodes, seeds 0 to 7 give 15, 12, 12, 16, 11, 16, 11 and 9 edges, so the
action and observation spaces change with the seed. Consequences: the
`random/*` rows in the benchmark average over different graphs, and an RL
agent cannot be trained or tested on "random" across seeds at all.

Fix: a separate `topology_seed` argument (default fixed). Effort: an hour.

### F4. Speed falls off with size; routing dominates
Grid, load scaled with size, equal-split policy, 300 steps:

| Nodes | Links | Steps/s | Share of step time in all-pairs Dijkstra |
|---|---|---|---|
| 9 | 12 | 509 | 24% |
| 16 | 24 | 225 | 34% |
| 25 | 40 | 149 | 43% |
| 36 | 60 | 119 | 51% |
| 49 | 84 | 68 | 59% |
| 64 | 112 | 42 | 65% |

At 49 nodes a 1M-step PPO run is about 4 hours of simulator time alone.
Caching paths when weights are unchanged helps only the heuristics (an RL
agent changes weights every step). Better: run single-source Dijkstra only
for sources that have new packets this step, or use
`scipy.sparse.csgraph.dijkstra`. These timings belong in the paper as they
are; "lightweight" is one of its claims. Effort: 1 day.

### F5. Synthetic traffic is uniform and stationary
Every packet picks source and destination uniformly; arrival rates are
constant Poisson; trace packets get a random QoS class. There is no hot
spot, no gravity structure and no time variation, so the word "volatile"
in the package description is not supported, and on a symmetric topology
the best static routing is close to shortest path. Add a gravity-model
matrix and a time-varying (diurnal or on-off) rate. SNDlib matrices
(PUBLICATION_READINESS item 6) supply realistic structure. Effort: 1 to 2
days.

### F6. Small items
* `bandwidth_budget` is counted in action units (`budget x number of
  links`), not in capacity. On the bottleneck scenario, half-capacity
  links cost the same budget as full ones. The docstring in
  `carla_gym/env.py:38` says "fraction of total link capacity"; correct it
  or weight by capacity.
* `RewardCalculator.normalize` names its last argument `delivered_count`,
  but `step.py` passes packets *resolved* (delivered plus expired). Rename.
* PPO and SAC use one fixed hyperparameter set (`gamma=0.95`, `lr=3e-4`).
  Reviewers ask; a small sweep on one scenario, reported in an appendix,
  answers it.

## 3. What the code already does well (keep it in the paper)
Tested invariants (134 tests including `check_env` and
`parallel_api_test`), train/validation/test seed separation, scripted
README numbers, delivered and dropped tables next to the reward, candid
limitations. This is above the norm for environment papers.

## 4. Revised order of work

| # | Work | Source | Before training? |
|---|---|---|---|
| 1 | F3 topology seed | this review | yes |
| 2 | F1 random tie-break and service order, fairness metric | this review | yes |
| 3 | F2 per-link queues | this review | yes |
| 4 | Rerun the baseline matrix and regenerate the README | follows 1-3 | yes |
| 5 | ECMP, inverse-capacity and LP-bound baselines | readiness item 3 | yes, so RL is compared to them |
| 6 | Training matrix (resume notes section 5) | readiness items 1-2 | the run itself |
| 7 | F4 faster routing, then the 25/36/49-node experiment | this review, readiness item 4 | no |
| 8 | F5 gravity and time-varying traffic; SNDlib data | this review, readiness item 6 | ideally yes |
| 9 | Rename, CI on Python 3.10-3.13, related-work table | readiness items 6-7 | no |

Items 1 to 5 are about four to five working days. They make the old
`results/grid_b08/` numbers obsolete; the README headline must be
regenerated from the new runs, not carried over. The venue advice in
`PUBLICATION_READINESS.md` section 6 stands.

## 5. Status after the Developer's fixes (15 Oct 2026, branch `fix/model-f1-f3`, version 0.4.0)

F1, F2 and F3 are fixed; the defaults are now per-link queues, random
equal-cost tie-breaks and random service order, with the graph drawn from
`topology_seed`. The 0.3.0-style behaviour stays available as options.
`tests/test_model_options.py` adds 29 tests (163 pass in total), including
the regression tests for QA findings Q1 and R2-2 (`docs/QA_PR5_REPORT.md`).
The table reflects the code after those fixes.
Measured with `qa/research/probe_after_fixes.py`, same seeds as above:

| Policy | Delivered, legacy options / new | Spread across sources, legacy / new | Unused allocated budget with traffic waiting, legacy / new |
|---|---|---|---|
| max_bandwidth | 0.769 / 0.953 | 0.304 / 0.028 | 14.3% / 2.8% |
| queue_proportional | 0.840 / 0.950 | 0.158 / 0.020 | 18.2% / 4.2% |
| congestion_aware | 0.867 / 0.954 | 0.052 / 0.022 | 20.1% / 4.2% |

The random topology now has 15 edges for every episode seed. Speed is about
the same as before: 624 steps/s at 9 nodes, 82 at 49, 58 at 64 on an idle
machine (`probe_after_fixes.log`); an earlier run of the same probe gave
574, 68 and 44, so read these as +/- 20%. F4 is still
open.

**Consequence for the experiments.** At grid 3x3, load x1, budget 0.8 every
baseline now delivers about 95% and they differ by under 0.01, so this is no
longer a congested setting and leaves little room for any policy to beat
another. The headline training scenario must be recalibrated (higher load
or a lower budget, chosen from the new baseline matrix) before the training
matrix is launched. The saved PPO and SAC models in `results/grid_b08/` were
trained on 0.3.0 and are kept as a record only.

## 6. Not checked
Python versions other than 3.14, Linux and macOS; the multi-agent
environment's dynamics beyond the shared step; whether the saved PPO model's
advantage changes under the fixes in F1 and F2 (it must be retrained to
know).

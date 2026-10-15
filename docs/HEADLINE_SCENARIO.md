# Headline training scenario for 0.4.0

> **Status, 15 Oct 2026: confirmed on 30 seeds.** The choice below was made
> on 10 seeds before the RNG fix from QA finding Q1 (`docs/QA_PR5_REPORT.md`).
> After that fix it failed criterion C5 on 10 seeds and the decision was
> reopened; a 30-seed rerun with rules fixed in advance selects the same
> scenario (last section). Tables in the earlier sections cite the pre-fix
> logs, kept as `*_preQ1` files.

Researcher role, 15 Oct 2026. Branch `fix/model-f1-f3` (0.4.0).

## Decision

**3x3 grid, synthetic traffic, load x1.25, bandwidth budget 0.6.**
Training flags: `--load-scale 1.25 --budget 0.6`. Proposed queue name:
`grid_l125_b06`.

It replaces `grid_b08` (load x1, budget 0.8), which under 0.4.0 is no longer
congested: every heuristic delivers about 95% of traffic
(`qa/research/probe_after_fixes.log`, `results/benchmark/qos.md`).

## How it was chosen

`qa/research/headline_sweep.py` ran 17 candidate scenarios x 5 policies on
**validation seeds 1000-1009**, so the test seeds (2000+) on which results
will be reported played no part in the choice. The criteria were fixed in
the script before it ran:

| # | Criterion |
|---|---|
| C1 | Congested but workable: the best heuristic delivers 65-88% |
| C2 | The bandwidth half matters: at least 3 points of delivered traffic between bandwidth-aware and equal split |
| C3 | The routing half matters: at least 3 points between load-aware and hop-count routing |
| C4 | Reward and delivered traffic rank the four heuristics the same way (Kendall tau >= 0.67) |
| C5 | Stable: 95% CI half-width of the best heuristic's reward <= 0.03 |

Five scenarios met all five (`qa/research/headline_sweep_preQ1.log`):

| Scenario | Best heuristic delivers | Bandwidth effect | Routing effect | CI |
|---|---|---|---|---|
| grid9, load 1.25, budget 0.6 | 73.9% | 3.8 pts | 3.1 pts | 0.027 |
| grid9, load 1.25, budget 0.7 | 83.3% | 4.6 pts | 4.4 pts | 0.020 |
| grid9, bottleneck, load 1, budget 0.6 | 71.3% | 11.5 pts | 20.8 pts | 0.026 |
| grid9, bottleneck, load 1, budget 0.8 | 84.6% | 9.5 pts | 21.9 pts | 0.023 |
| grid9, bottleneck, load 1.25, budget 0.8 | 72.6% | 9.2 pts | 20.4 pts | 0.021 |

The bottleneck scenarios are dominated by one engineered fault and are
already in the plan as their own scenario (`grid_bottleneck_b08`, which is
the fourth row and qualifies). The headline should be the plain grid.

Between the two plain-grid finalists the pre-set tie-break (smaller
scenario) does not decide, since both are 9 nodes. **I broke the tie with a
criterion added after seeing the sweep**, and say so: whether controlling
both halves beats controlling either one alone. That is what the planned
bandwidth-only and routing-only ablation tests, so the headline scenario
must give it something to find. Paired effects over the validation seeds,
delivered traffic in points, mean +/- 95% CI
(`qa/research/headline_effects_preQ1.log`):

| Effect | load 1.25, budget 0.6 | load 1.25, budget 0.7 |
|---|---|---|
| bandwidth, with hop-count routing (queue_prop - max_bw) | +3.76 +/- 0.66 | +4.60 +/- 0.57 |
| bandwidth, with load-aware routing (congestion - load_aware) | **+2.06 +/- 1.03** | +0.58 +/- 0.58 |
| routing, with equal split (load_aware - max_bw) | +3.09 +/- 1.22 | +4.42 +/- 0.97 |
| routing, with proportional bandwidth (congestion - queue_prop) | **+1.39 +/- 0.46** | +0.40 +/- 0.67 |

At budget 0.6 the two halves add up: the heuristic using both beats each
single-half heuristic. At budget 0.7 they are substitutes, and combining
them adds nothing measurable. Budget 0.6 also leaves more room above the
heuristics (best delivers 73.9% against 83.3%).

Weak points to keep in view: its routing effect (3.1 pts) and its CI (0.027)
sit close to the thresholds; and how far any policy can get above 73.9% is
unknown until the multi-commodity-flow bound (readiness plan item 3b) exists.

## Consequences for the training queue (`scripts/run_training_matrix.py` on `feat/rl-scenarios`)

| Queue entry | Status under 0.4.0 |
|---|---|
| `grid_b08` | Replace with `grid_l125_b06` |
| `grid_b08_bandwidth_only`, `grid_b08_routing_only` | Move to the headline: `grid_l125_b06_bandwidth_only`, `..._routing_only` |
| `grid_bottleneck_b08` | Keep; meets all five criteria |
| `pcap40_grid16_b05` | Keep; heuristics separate clearly (delivered 57% to 76%, test-seed matrix) |
| `ring_b08` | Does not qualify as it stands: at load x1, budget 0.8 all heuristics are tied in the test-seed matrix. A second topology needs the same sweep |
| `grid_load2_b08` | Does not qualify: heuristics within 2.5 points (56-59% delivered). Keep only as a labelled overload case, or drop to save compute |

The test-seed matrix figures in this table are descriptive only; any
replacement for `ring_b08` should be chosen on validation seeds as above.

## Second topology: none qualifies (15 Oct 2026)

`python qa/research/headline_sweep.py second_topology` scored 29 cells on
validation seeds 1000-1009: ring (9 links), the fixed random graph
(`topology_seed` 0, 15 links), full mesh (36 links) and, for comparison,
the 4x4 grid. Criteria fixed before it ran: C1-C5 as above plus

* **C6** controlling both halves beats each half alone (congestion_aware
  minus load_aware_routing, and minus queue_proportional, paired over seeds,
  95% CI lower bound > 0).

Tie-breaks were set as non-grid first, then smallest action space. Star was
excluded: it is a tree, so routing weights cannot change a path.

**Result: no cell met all six** (`qa/research/second_topology_sweep_preQ1.log`).

| Topology | Best cells | What fails |
|---|---|---|
| ring9 | at most 2 of 6 | Routing has no leverage (at most 0.8 points); with two ways round a ring that is expected |
| random9 | load 1.25 / budget 0.5 and load 1.5 / budget 0.6 meet C1-C5 | C6. Bandwidth matters a lot (16 to 20 points), but load-aware routing helps little with equal split (+2.4 and +1.0 points) and **hurts** with proportional bandwidth (-3.5 and -3.4 points, from the per-policy table in the log) |
| mesh9 | 4 of 6 | C6 always, plus C1 or C3; same pattern as random9 |
| grid16 | 4 of 6 | C4 and C6 in the best cells |

Two defects in my own criteria, found on reading these results. Neither
changes the outcome, and the script is left as it ran:

* **C2 and C3 use absolute differences**, so a routing heuristic that
  *hurts* counts as "routing matters". random9 passes C3 only that way.
  In the headline sweep every chosen effect was positive, so the headline
  choice is unaffected.
* **C4's threshold 0.67 rejects tau = 2/3** (one discordant pair of six),
  which I meant to allow. With 2/3 allowed, grid16 load 1.5 / budget 0.6
  and load 1.75 / budget 0.7 would meet C1-C5; both still fail C6.

### What this means

Off the 3x3 grid, the routing-aware heuristics are neutral or harmful. The
sweep cannot tell whether that is because **routing has little leverage on
those graphs** or because **the heuristic is a poor router**; both read the
same through criterion C3. Two things would separate them:

1. the multi-commodity-flow bound (readiness plan item 3b), computed once
   with routing fixed to shortest paths and once with free multipath
   routing. The gap is the leverage routing has, independent of any policy;
2. a trained routing-only agent on that graph.

### Recommendation

Do not hold up the training queue for this. Put **one** second-topology
scenario in the queue, labelled as *bandwidth-dominated*, not as a
replication of the headline:

* **`random9_l15_b06`: random graph, `--topology random --load-scale 1.5
  --budget 0.6`.** It meets C1-C5 (C3 only in the absolute sense above).
  Between it and load 1.25 / budget 0.5, which are nearly identical, the
  pre-set tie-breaks do not decide; I chose budget 0.6 after seeing the
  results, so that the headline and this scenario differ in topology and
  load but not in budget.
* Train PPO and SAC, and **add a routing-only PPO ablation** there (5 runs).
  If a learned router gains on this graph where the heuristic loses, that is
  a result in its own right; if it does not, the paper reports that the
  joint-control benefit is topology-dependent.
* Drop `ring_b08`. Keep the 4x4 grid for the scale experiment, not as a
  second topology.

Paper wording this supports today: *on the 3x3 grid the two action halves
are complementary; on the random graph tested, bandwidth allocation carries
most of the benefit.* Not supported: any claim that joint control helps in
general.

## Rerun after the RNG fix (Developer, 15 Oct 2026): decision reopened

QA finding Q1 made the network's random stream independent of the traffic
stream. That changes every episode, so both sweeps were rerun unchanged on
the same validation seeds (`qa/research/headline_sweep.log`,
`headline_effects.log`, `second_topology_sweep.log`,
`second_topology_effects.log`).

**Headline sweep.** The chosen scenario no longer meets the criteria:

| Scenario | Before the fix | After the fix |
|---|---|---|
| grid9, load 1.25, budget 0.6 (chosen) | all five met; CI 0.027 | **fails C5: CI 0.032** (limit 0.03) |
| grid9, load 1.25, budget 0.7 | all five met | fails C4 (tau 2/3; see the C4 defect above) |
| grid9, load 1, budget 0.5 | fails C5 (CI 0.034) | **all five met**: best heuristic delivers 77.4%, bandwidth 5.1 pts, routing 3.4 pts, CI 0.028 |

After the fix, grid9 load 1 / budget 0.5 is the only plain-grid cell that
meets all five, so the pre-set rules pick it without any tie-break. It also
passes the complementarity test used to break the earlier tie: congestion_aware
beats load_aware_routing by +2.46 +/- 0.55 points and queue_proportional by
+0.79 +/- 0.37. The three bottleneck cells that qualified before still
qualify.

**Second-topology sweep.** Conclusion unchanged: no cell meets all six.
random9 load 1.5 / budget 0.6 still meets C1-C5 (fails C6). Two grid16
cells now reach five of six (load 1.75 / budget 0.6 fails C3; load 1.75 /
budget 0.7 fails C4).

**What this shows.** Which plain-grid cell passes C5 flips with a change to
the random stream alone. Their CIs sit within a few thousandths of the 0.03
limit, and a CI estimated from 10 seeds is itself noisy. The headline choice
is therefore not robust at this sample size, whichever cell is picked.

**Not decided here.** Choosing the headline is the Researcher's call. The
training queue on `feat/rl-scenarios` still names `grid_l125_b06` and must not
be launched until this is settled. A suggested way to settle it: rerun the
plain-grid cells on 30 validation seeds (1000-1029), so the stability
criterion is not decided by noise, then apply the same rules.

## Settled on 30 validation seeds (Researcher, 15 Oct 2026)

`python qa/research/headline_settle30.py` reran the nine plain 3x3 grid
cells on validation seeds 1000-1029 (test seeds unused), after the Q1 fix.
Rules fixed in the script before it ran, with the two documented criteria
defects corrected:

* C1 best heuristic delivers 65-88%;
* C2, C3 now **signed**: the bandwidth-aware (routing-aware) heuristic must
  *beat* the blind one by at least 3 points;
* C4 Kendall tau >= 2/3;
* C5 CI half-width of the best heuristic's reward <= 0.03;
* C6 congestion_aware beats each single-half heuristic (paired, CI lower
  bound > 0);
* tie-break: the largest weaker C6 lower bound, then the lower load.

Three cells met all six (`qa/research/headline_settle30.log`):

| Cell | Weaker C6 lower bound | Both halves vs routing only | Both halves vs bandwidth only | Best delivers | CI |
|---|---|---|---|---|---|
| **grid9, load 1.25, budget 0.6** | **0.90** | +1.90 +/- 0.35 | +1.26 +/- 0.37 | 73.7% | 0.014 |
| grid9, load 1, budget 0.5 | 0.87 | +2.68 +/- 0.28 | +1.11 +/- 0.24 | 78.4% | 0.021 |
| grid9, load 1.25, budget 0.7 | 0.50 | +0.92 +/- 0.37 | +0.92 +/- 0.41 | 83.6% | 0.010 |

**Decision: the headline stays grid9, load x1.25, budget 0.6
(`grid_l125_b06`).** The training queue needs no change.

* Its C5 failure on 10 seeds (0.032) was sampling noise: on 30 seeds its CI
  is 0.014, well inside the limit.
* The margin over load 1 / budget 0.5 is small (0.90 against 0.87). Both
  are sound headlines; the rule decides, and the paper should not present
  the choice as sharper than that.
* Its effects at 30 seeds: bandwidth 4.3 points, routing 3.7 points, and the
  two halves complementary by +1.9 and +1.3 points.

Not rerun on 30 seeds: the bottleneck and second-topology cells. Their
conclusions rest on larger margins (bottleneck effects of 9 to 22 points;
no second-topology cell meets C6 on either 10-seed run).

*Note on the sweep logs.* All sweeps above ran before the fix for QA finding
R2-2, which gave the `random` baseline its own random stream. Only the
`random_delivered_%` column of the sweep logs is affected; no criterion uses
the random policy, so no decision here depends on it.

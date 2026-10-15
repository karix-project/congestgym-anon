# RL results on the 0.4.0 simulator: first reading

Researcher role, 15 Oct 2026. Not yet checked by Tester/QA.

Data: the medium training matrix (PPO 500k steps, SAC 100k, 5 training seeds
each, test seeds 2000-2009), `results/rl/`. Agent and deterministic-baseline
rewards were re-evaluated by `scripts/evaluate_saved.py` and match each
run's `final_eval.csv` exactly. Every number below is from
`qa/research/rl_results_analysis.log` or a `results/rl/*/qos.md` table.

## 1. Short answer

| Scenario | PPO vs best heuristic, delivered traffic | PPO vs best heuristic, reward | Verdict |
|---|---|---|---|
| Headline: 3x3 grid, load x1.25, budget 0.6 | -0.6 +/- 0.6 pts (wins 16/50) | -0.062 +/- 0.016 (wins 0/50) | **RL does not beat the best heuristic** |
| Bottleneck: centre node at half capacity | **+9.7 +/- 0.3 pts** (50/50) | +0.173 +/- 0.016 (50/50) | RL wins clearly |
| pcap trace, 4x4 grid | **+4.0 +/- 0.7 pts** (50/50) | +0.056 +/- 0.006 (50/50) | RL wins clearly |
| Random graph (bandwidth-dominated) | **+1.6 +/- 0.4 pts** (47/50) | +0.025 +/- 0.008 (44/50) | RL wins narrowly |

Paired on test seeds; the interval is over the 5 training seeds; "wins"
counts (training seed, test seed) pairs out of 50. The best heuristic is
chosen per scenario and per metric. SAC is below PPO everywhere and beats
the best heuristic only on the bottleneck.

**The 0.3.0 headline result does not survive the simulator fixes.** In 0.3.0
PPO beat the best heuristic by +0.090 in reward on the old headline, winning
50 of 50 pairs. On the 0.4.0 headline PPO loses on reward in all 50 pairs.
Part of the old advantage was earned against a simulator that wasted 16 to
20% of allocated bandwidth (`docs/CODE_REVIEW_FOR_PUBLICATION.md`, F2).

## 2. Findings

### 2.1 Where RL wins, and where it does not
The learned policy wins where the network has structure a fixed rule does
not exploit: a degraded node (+9.7 points delivered), a trace that
concentrates load on a few pairs (+4.0), an irregular graph (+1.6). It does
not win on the symmetric grid with uniform traffic, where congestion_aware
is already strong.

On the headline, PPO delivers about as much traffic as congestion_aware
(73.5% against 74.1%; difference -0.6 +/- 0.6 points), but its delivered
packets take longer (5.82 against 5.16 steps) and miss more deadlines
(0.370 against 0.330 per delivered packet). The reward's delay and deadline
terms make that a clear loss in reward (-0.062).

It is not a matter of longer paths (`qa/research/headline_paths.log`): PPO's
delivered packets travel 1.868 +/- 0.006 hops on average, congestion_aware's
1.872 +/- 0.007, and every policy delivers every packet on a shortest path
(stretch 1.000). Since delay is hops plus steps spent waiting, PPO's extra
delay is queueing delay, which points at its bandwidth allocation, not its
routing.

### 2.2 The ablations do not show the joint-control benefit for the learner
Headline, full-control PPO minus one-half PPO (paired):

| Comparison | Reward | Delivered |
|---|---|---|
| full minus bandwidth-only | -0.013 +/- 0.015 | +0.5 +/- 0.6 pts |
| full minus routing-only | +0.038 +/- 0.020 | +1.3 +/- 0.6 pts |

Full control beats routing-only, but it does **not** beat bandwidth-only:
on reward, bandwidth-only PPO is slightly ahead, within noise. The
heuristics in this scenario showed the two halves to be complementary
(`docs/HEADLINE_SCENARIO.md`: congestion_aware beats each single-half
heuristic). PPO at 500k steps does not exploit that: given both halves, it
does about as well as with bandwidth alone.

Random graph: full PPO beats routing-only PPO by +15.0 +/- 0.4 points,
consistent with the bandwidth-dominated label. Routing-only PPO (66.3%)
beats the load-aware routing heuristic (64.8%) and equal split (63.3%), so a
learned router does better than the heuristic router there, but routing alone
recovers little. There is no bandwidth-only ablation on this graph, so
whether learned routing adds anything on top of bandwidth here is not
measured.

### 2.3 Convergence
Change in validation reward over the last fifth of training (400k to 500k
for PPO, 80k to 100k for SAC), mean +/- 95% CI over training seeds:

* **Flat:** headline PPO (+0.011 +/- 0.018), pcap PPO (-0.000 +/- 0.009), all
  three ablations (+0.002 to +0.007).
* **Still rising slightly:** bottleneck PPO (+0.032 +/- 0.026) and
  random-graph PPO (+0.017 +/- 0.011).
* **SAC:** changes of -0.003 to +0.017 with intervals of +/- 0.025 to 0.060
  in all four scenarios, too wide to call either way.

So more PPO steps would probably not change the headline verdict, but could
widen the wins on the bottleneck and the random graph. Untuned
hyperparameters (one set, gamma 0.95) remain a more likely lever than steps
on the headline.

### 2.4 On the headline grid, routing never takes a detour
In the same measurement, no policy sends any delivered packet over a path
longer than the shortest one: not the load-aware heuristics, not PPO. On the
3x3 grid, every routing effect in this report (heuristic or learned) comes
from choosing among equal-length shortest paths, not from steering traffic
around congestion on longer routes. Say so wherever the paper discusses
routing on this scenario. Whether detours occur on the bottleneck or the
random graph was not measured.

### 2.5 Reward and delivered traffic
Kendall tau between the reward ranking and the delivered ranking of all
policies except random and min_bandwidth: bottleneck 1.00, random graph
0.87, pcap 0.73, headline 0.60. The headline disagreement is the PPO effect
in 2.1 (fourth on reward, second on delivered). Report delivered, delay and
deadline tables next to the reward, as the README already advises.

### 2.6 The flow upper bound, and why it cannot settle the headline
`scripts/flow_bound.py` (`results/bounds/training.md`) gives a fluid upper
bound on the delivered fraction: the expected traffic matrix, link capacities
and the budget, with the allocation chosen optimally. No policy exceeds it
by more than 0.3 points in any of 35 scenarios (that excess is backlog
carried into the window), so it behaves as a bound.

| Scenario | Flow bound | Bound's allocation as a fixed policy (scaled to the full budget) | Best heuristic | PPO |
|---|---|---|---|---|
| Headline grid | 90.3 | 66.8 +/- 1.5 | 74.1 | 73.5 |
| Bottleneck | 100.0 | 62.8 +/- 0.8 | 85.0 | 94.6 |
| pcap trace | 100.0 | 60.7 +/- 1.5 | 75.2 | 79.1 |
| Random graph | 99.4 | 71.1 +/- 1.6 | 79.7 | 81.3 |

Delivered %, test seeds 2000-2009 (`qa/research/lp_static_policy.log`).

* **The bound is loose.** Playing its own optimal allocation as a fixed policy
  delivers less than every adaptive policy, and on the headline less than
  equal split (68.5%). An allocation that is optimal for the mean load is a
  poor policy, because what the heuristics and PPO gain comes from reacting
  to queues as they form, which the fluid model leaves out.
* **The bound also assumes a choice no policy has.** On the headline its
  optimum is exactly a fractional knapsack (`qa/retest/qa_bounds.log`): it
  serves the shortest source-destination pairs first and drops the longest
  pairs entirely. In this environment no policy chooses which traffic to
  drop; admission depends only on buffer space. For comparison only, giving
  every pair the same delivered fraction under the same budget gives about
  82.7% (`docs/QA_BOUNDS_REPORT.md`, QB-2). That figure is **not** an upper
  bound, since a policy may favour short pairs indirectly; it shows that
  several of the 16 points between 74.1% and 90.3% come from this effect
  alone.
* **So the headline question stays open.** The optimum lies somewhere between
  74.1% (the best policy found) and 90.3% (the bound); this does not show
  whether congestion_aware is near-optimal there, or how much room PPO left.
  The room below the bound is an upper limit on what any policy could gain,
  not a measure of headroom, and the paper must not present it as one.
  (`results/bounds/training.md` had a "share of the gap" column; it was
  removed for this reason.)
* **Detours add no capacity.** The bound is the same with free multipath
  routing and with routing restricted to shortest paths, in all 35
  scenarios. A likely reason is that the budget can be moved between links,
  so spending it on the shortest routes was at least as good here; per-link
  limits did bind on 1 to 2 links in some LP solutions without changing
  this. With 2.4, this means every routing
  gain in this simulator is dynamic (spreading load across equal-length
  paths as queues change, avoiding buffer overflow), not a matter of
  capacity. That is how the paper should present the routing action.

## 3. What the paper can claim from this

Supported:
* A learned policy (PPO) beats every rule-based baseline in 3 of the 4
  scenarios, by 1.6 to 9.7 points of delivered traffic, most in the
  bottleneck scenario.
* On a symmetric grid with uniform traffic, a simple congestion-aware rule
  matches PPO on delivered traffic and beats it on delay and deadlines.
* PPO outperforms SAC at the budgets used (SAC had a fifth of PPO's steps;
  state the budgets).
* No policy exceeds the fluid flow bound; on the bottleneck PPO reaches
  94.6% of a 100% bound.
* Routing gains are dynamic: detours add no capacity in any scenario tested.

Not supported:
* "RL beats heuristics" without qualification.
* Any claim that the learner benefits from controlling both action halves.
  The heuristics show complementarity on the headline; the learner does not.
* The 0.3.0 numbers. They must not appear in the paper as current results.
* That the heuristic is near-optimal on the headline, or that PPO leaves a
  measured amount of headroom there. The fluid bound is too loose to say.

## 4. Recommended next steps

1. **Done: the flow upper bound** (2.6). It is too loose to decide the
   headline. A tighter reference needs to account for queueing; the
   cheapest options are a stronger tuned policy (step 2) or a search over
   the heuristics' own parameters on validation seeds.
2. **A small PPO hyperparameter sweep on the headline** (gamma 0.95 / 0.99,
   learning rate, entropy coefficient), on validation seeds, before
   concluding that PPO cannot match congestion_aware there.
3. **Complete the ablation grid** at small cost: bandwidth-only PPO on the
   random graph, and both ablations on the bottleneck, where routing matters
   most (heuristic routing effect 21 points). If the joint-control benefit
   exists for a learner anywhere, the bottleneck is the likeliest place.
4. **Longer PPO runs only where curves still rise** (bottleneck, random
   graph), not across the matrix.
5. Keep the headline as chosen. Swapping it after seeing that RL loses there
   would be selection on the outcome.

## Changes after QA (15 Oct 2026)
QA findings QR-1 and QR-2 (`docs/QA_RL_RESULTS_REPORT.md`): the SAC
convergence wording is made consistent (2.3), and the unmeasured "worse order
or longer paths" is replaced by a measurement (2.1, new 2.4,
`qa/research/headline_paths.py` / `.log`).
Section 2.6 (flow bound and the static-allocation check) added the same day,
after `scripts/flow_bound.py` was written; not yet checked by Tester/QA.
QA findings QB-1 and QB-2 (`docs/QA_BOUNDS_REPORT.md`) applied to 2.6: the
free = shortest explanation is stated as likely, not as a result, and the
knapsack source of the bound's looseness is added.

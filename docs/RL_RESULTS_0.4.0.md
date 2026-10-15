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
terms make that a clear loss in reward (-0.062). So PPO matches the
throughput but serves traffic in a worse order or over longer paths.

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
  three ablations (+0.002 to +0.007), pcap SAC.
* **Still rising slightly:** bottleneck PPO (+0.032 +/- 0.026) and
  random-graph PPO (+0.017 +/- 0.011).
* **SAC:** intervals of +/- 0.025 to 0.060, too wide to call.

So more PPO steps would probably not change the headline verdict, but could
widen the wins on the bottleneck and the random graph. Untuned
hyperparameters (one set, gamma 0.95) remain a more likely lever than steps
on the headline.

### 2.4 Reward and delivered traffic
Kendall tau between the reward ranking and the delivered ranking of all
policies except random and min_bandwidth: bottleneck 1.00, random graph
0.87, pcap 0.73, headline 0.60. The headline disagreement is the PPO effect
in 2.1 (fourth on reward, second on delivered). Report delivered, delay and
deadline tables next to the reward, as the README already advises.

## 3. What the paper can claim from this

Supported:
* A learned policy (PPO) beats every rule-based baseline in 3 of the 4
  scenarios, by 1.6 to 9.7 points of delivered traffic, most in the
  bottleneck scenario.
* On a symmetric grid with uniform traffic, a simple congestion-aware rule
  matches PPO on delivered traffic and beats it on delay and deadlines.
* PPO outperforms SAC at the budgets used (SAC had a fifth of PPO's steps;
  state the budgets).

Not supported:
* "RL beats heuristics" without qualification.
* Any claim that the learner benefits from controlling both action halves.
  The heuristics show complementarity on the headline; the learner does not.
* The 0.3.0 numbers. They must not appear in the paper as current results.

## 4. Recommended next steps

1. **The flow upper bound** (readiness plan item 3b) on the headline. If the
   bound is close to 74%, congestion_aware is near-optimal there and the
   headline verdict is a property of the scenario, not of the learner. This
   decides how to present 2.1.
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

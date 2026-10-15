# Next experiments: plan and decision rules

Researcher role, 15 Oct 2026, after PR #6 (0.4.0 training matrix) was
merged. Nothing here has been run. Rules are fixed before any run, as for the
scenario choice (`docs/HEADLINE_SCENARIO.md`), so the results cannot steer
how they are read.

## 1. Questions the current results leave open

From `docs/RL_RESULTS_0.4.0.md`:

| # | Question | Why it matters for a paper |
|---|---|---|
| Q-A | Does a tuned PPO match or beat congestion_aware on the headline? | The headline loss could be the scenario or one untuned hyperparameter set; a reviewer will ask which |
| Q-B | Are the baselines as strong as they can be? | If RL is tuned and the heuristics are not, every RL win is open to "unfair comparison" |
| Q-C | Does joint control help a learner anywhere? | On the headline it did not; the bottleneck, where routing matters most for the heuristics (about 21 points), is the likeliest place |
| Q-D | Does learned routing add anything on top of bandwidth on the random graph? | Only routing-only was trained there; bandwidth-only is missing |
| Q-E | Do learned policies ever take detours on the bottleneck or the random graph? | On the headline no policy ever does; the fluid bound says detours add no capacity anywhere |

Out of scope here: longer training (the headline curve is flat; see 2.3 of the
reading), SAC tuning, new scenarios.

## 2. Experiments

### E1. Tune the heuristics on validation seeds (answers Q-B; no RL, cheap)
`queue_proportional` and `congestion_aware` have hard-coded constants: the
minimum bandwidth weight (0.05) and the routing cost `0.1 + 0.9 x
congestion`. Grid search them on the four training scenarios, validation
seeds 1000-1009:

* bandwidth floor in {0.0, 0.05, 0.1, 0.2};
* routing cost base in {0.0, 0.1, 0.3}, with slope 1 - base;
* congestion signal: max(pending, endpoint occupancy) as now, or pending only.

That is 4 x 3 x 2 = 24 settings per scenario, 10 episodes each: about 960
episodes, under 15 minutes on 10 cores. **Rule:** per scenario, the setting
with the highest mean validation **reward** becomes `tuned_congestion_aware`
(and the best bandwidth-only setting `tuned_queue_proportional`). Both are
then reported on the test seeds next to the untuned ones. No setting is
chosen on test seeds.

Needs (Developer): the constants as keyword arguments of the two policies,
defaults unchanged.

### E2. PPO hyperparameter sweep on the headline (answers Q-A)
Fixed grid, 8 configurations, 3 training seeds each, 500k steps:

* gamma in {0.95, 0.99};
* learning rate in {3e-4, 1e-4};
* entropy coefficient in {0.0, 0.01}.

`n_steps`, batch size and network size stay as now, to keep the grid small.
24 runs. **Rule:** the configuration with the highest mean held-out reward at
500k on validation seeds (the existing learning curve, seeds 1000-1004),
averaged over its 3 seeds, is chosen. It then gets 2 more training seeds (5
in total) and is evaluated once on test seeds 2000-2009, with the same paired
protocol as before, against untuned **and** tuned congestion_aware (E1).

Pre-registered reading:
* tuned PPO minus best rule-based policy, reward, 95% CI entirely above 0:
  "tuning closes the gap";
* CI includes 0: "tuned PPO matches the best rule-based policy";
* CI entirely below 0: "tuning does not close the gap; on the symmetric grid
  a congestion-aware rule is at least as good as PPO".

All three outcomes are publishable; none is hoped for.

Needs (Developer): `--gamma`, `--lr`, `--ent-coef` on `train_agent.py`
(defaults unchanged), and an `--hp-sweep` mode or a second queue in
`run_training_matrix.py`.

### E3. Complete the ablations (answers Q-C, Q-D)
With the current hyperparameters, PPO, 5 seeds, 500k steps:

* `grid_bottleneck_b08_bandwidth_only`, `grid_bottleneck_b08_routing_only`;
* `random9_l15_b06_bandwidth_only`.

15 runs. **Rule, per scenario:** full-control PPO minus each one-half PPO,
paired over (training seed, test seed), reward and delivered. Joint control
"helps the learner" only where full control beats **both** halves with the
95% CI above 0.

If E2 changes the hyperparameters, E3 is run with the current ones anyway,
so that it stays comparable with the existing full-control runs.

### E4. Detours on the bottleneck and the random graph (answers Q-E; analysis only)
Run `qa/research/headline_paths.py`'s measurement (hops, stretch, share of
packets on a longer-than-shortest path) on the existing bottleneck and
random-graph models and baselines. Stretch is measured against hop-count
shortest paths for the random graph, as on the grid. Minutes, no training.

## 3. Compute

From the last matrix (10 runs at a time on this laptop): PPO on a 9-node
scenario took 44 to 73 minutes per 500k-step run, slowing later in the day.

| Tier | Contents | Runs | Rough wall time |
|---|---|---|---|
| Full (recommended) | E1 + E2 (24 + 2) + E3 (15) + E4 | 41 | about 5 to 6 hours |
| Without E3 | E1 + E2 + E4 | 26 | about 3.5 hours |
| Cheapest | E1 + E4 only | 0 (no training) | under 30 minutes |

E1 and E4 should run first either way: E1 is needed before E2 is read, and
both are cheap.

## 4. What would change in the paper

* E1 removes the "untuned baselines" objection whichever way it comes out.
* E2 decides how the headline is described (see the pre-registered reading).
* E3 decides whether "joint control helps a learner" can be claimed anywhere,
  and if so where.
* E4 completes the routing story: either detours never happen (routing is
  tie-choice only, everywhere tested) or they do on irregular graphs.

## 5. Order of work
1. Developer: expose the heuristic constants and the PPO hyperparameters
   (defaults unchanged, tests that defaults reproduce existing results
   exactly); add the E2 and E3 queues.
2. Researcher: E1 and E4, then confirm the E2 grid before launch.
3. User: choose the compute tier.
4. Run E2 (and E3), then Researcher, then Tester/QA.

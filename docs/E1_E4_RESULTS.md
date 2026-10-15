# Experiments E1 and E4: results

Researcher role, 15 Oct 2026. Plan and rules: `docs/NEXT_EXPERIMENTS.md`.
No training; existing PPO models from `results/rl/`. Not yet checked by
Tester/QA.

## E1. Tuned rule-based policies

`qa/research/e1_tune_heuristics.py` (`.log`). Grid search on validation seeds
1000-1009 (24 settings of congestion_aware, 4 of queue_proportional, per
scenario); the setting with the highest mean validation reward was chosen,
then evaluated once on test seeds 2000-2009. Running the script twice gave
identical choices.

### What tuning changed (test seeds, paired with the untuned policy)

| Scenario | congestion_aware: chosen setting | Reward, tuned minus untuned | Delivered, tuned minus untuned |
|---|---|---|---|
| Headline grid | floor 0, base 0, signal max | -0.006 +/- 0.011 | -0.15 +/- 0.69 pts |
| Bottleneck | floor 0.2, base 0.3 | **+0.031 +/- 0.011** | +0.83 +/- 0.59 pts |
| pcap trace | floor 0.1, base 0.3 | **+0.109 +/- 0.011** | **+1.66 +/- 0.88 pts** |
| Random graph | floor 0.1, base 0.3 | **+0.057 +/- 0.014** | **+1.77 +/- 0.54 pts** |

queue_proportional moved by at most 0.012 in reward in any scenario.

* **On the headline, the published congestion_aware is already as good as
  this family gets.** The setting chosen on validation does no better on test
  (difference within noise), so the default constants were not leaving
  anything on the table there.
* **Elsewhere, tuning helps congestion_aware, always in the same direction:**
  a higher routing-cost base (0.3), which makes routing less sensitive to
  congestion and so takes fewer detours (see E4).

### Do PPO's wins survive the strongest rule-based policy?
`qa/research/e1_ppo_vs_best.py` (`.log`). Comparator: the best of the four
untuned heuristics and the two tuned ones, chosen per scenario and per
metric **on the test seeds** (which favours the rule-based side). PPO minus
comparator, paired, CI over 5 training seeds:

| Scenario | Reward | Delivered | Before E1 (delivered) |
|---|---|---|---|
| Headline grid | -0.062 +/- 0.016 (0/50) | -0.6 +/- 0.6 pts (16/50) | -0.6 pts |
| Bottleneck | **+0.172 +/- 0.016** (50/50) | **+9.7 +/- 0.3 pts** (50/50) | +9.7 pts |
| pcap trace | **+0.044 +/- 0.006** (50/50) | **+2.3 +/- 0.7 pts** (49/50) | +4.0 pts |
| Random graph | **+0.025 +/- 0.008** (44/50) | **+1.6 +/- 0.4 pts** (47/50) | +1.6 pts |

**PPO's three wins survive tuned baselines.** The pcap margin shrinks from
+4.0 to +2.3 points of delivered traffic; the others are unchanged. The
headline result is unchanged: the strongest rule-based policy there is still
the untuned congestion_aware.

## E4. Detours on the bottleneck and the random graph

`qa/research/e4_detours.py` (`.log`). Packets delivered on test seeds;
stretch = hops / hop-count shortest-path length.

| Scenario | Policy | Packets on a detour | Mean stretch |
|---|---|---|---|
| Bottleneck | load_aware_routing | 0.61 +/- 0.20% | 1.006 |
| Bottleneck | congestion_aware | 0.34 +/- 0.12% | 1.003 |
| Bottleneck | PPO | 0.76 +/- 0.31% | 1.008 |
| Random graph | load_aware_routing | **6.29 +/- 1.03%** | 1.044 |
| Random graph | congestion_aware | **7.84 +/- 0.34%** | 1.053 |
| Random graph | PPO | 0.43 +/- 0.25% | 1.002 |
| Random graph | PPO, routing only | 1.13 +/- 0.33% | 1.006 |

Bandwidth-only policies never detour (routing weights are uniform).

* **Bottleneck:** detours are rare for every policy (under 1% of packets).
  The routing-aware heuristics' large advantage there (about 21 points over
  equal split) therefore comes from choosing among equal-length paths that
  avoid the degraded node, not from going the long way round.
* **Random graph:** the routing-aware heuristics detour 6 to 8% of packets,
  and they also deliver less there than bandwidth-only queue_proportional.
  PPO almost never detours, consistent with the fluid bound's finding that
  detours add no capacity. Fewer detours go with better delivery: tuning cut
  congestion_aware's detours from 7.8% to 2.6% of packets
  (`qa/retest/qa_e1_tuned_detours.log`) and raised its delivered traffic by
  1.8 points. This is a correlation: tuning also changed how the heuristic
  chooses among equal-length paths, so detours are not shown to be the
  cause.

## What this changes

1. **The comparison with the rule-based policies is now fair in both
   directions.** RL wins in three scenarios against baselines tuned on
   validation seeds; the paper can say so.
2. **The routing story for the scenarios tested:** no case was found where
   detours help. On the grids, routing gains come from choosing among
   equal-length paths; on the random graph, the heuristics that detour most
   deliver least, and the learner rarely detours.
3. **For E2** (the PPO sweep on the headline), the comparator stays the
   untuned congestion_aware: it is the strongest rule-based policy there on
   both reward and delivered traffic.

## Changes after QA (15 Oct 2026)
QA finding QE-1 (`docs/QA_E1_E4_REPORT.md`): the causal and intent wording
in E4 and in "What this changes" is replaced by what was measured, including
the tuned heuristic's detour share measured by QA (2.56% against 7.84%).

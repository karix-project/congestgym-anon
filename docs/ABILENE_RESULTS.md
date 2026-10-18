# Abilene scenario: results

Researcher role, 18 Oct 2026. Scenario `abilene_l120_b08` (SNDlib Abilene
topology and one-week trace, mean load 120, budget 0.8, random episode
starts; `docs/ABILENE_SCENARIO.md`). PPO 500k steps, SAC 100k, 5 training
seeds each; test seeds 2000-2009. Numbers from
`qa/research/abilene_results.py` (`.log`) and
`results/rl/abilene_l120_b08/qos.md`. Not yet checked by Tester/QA.

## 1. Short answer

| | Delivered | Reward minus congestion_aware (paired) | Delivered minus congestion_aware (paired) |
|---|---|---|---|
| congestion_aware (best rule-based) | 80.7% | - | - |
| SAC | 78.5% | -0.037 +/- 0.049 (wins 16/50) | -2.3 +/- 2.3 pts (wins 11/50) |
| PPO | 75.4% | **-0.133 +/- 0.097 (wins 3/50)** | **-5.3 +/- 3.3 pts (wins 3/50)** |

**On real Abilene topology and traffic, no learned policy beats the
congestion-aware rule.** SAC matches it within noise, though the delivered
interval only just includes zero (-2.3 +/- 2.3 points); PPO is clearly below it.
The rule's lead over the next rule-based policy (load_aware_routing) is
+0.138 +/- 0.039 reward, +5.9 +/- 1.6 points.

Intervals are paired over test seeds: the time of day an episode samples
moves every policy's numbers together, so unpaired intervals (up to +/- 9
points delivered for the rule-based policies) overstate the uncertainty of
the comparison.

## 2. Findings

### 2.1 PPO was still learning
Validation reward over the last fifth of training (400k to 500k steps):
**+0.117 +/- 0.078**, with the last three checkpoints at +0.047, +0.068 and
+0.081. That is the clearest still-rising curve in the project (on the
synthetic scenarios the same change was at most +0.032). SAC's change was
+0.013 +/- 0.073 (noisy, no clear trend). So PPO's loss on this scenario
may partly be a matter of training budget; a longer PPO run is the obvious
next experiment here. SAC, at a fifth of PPO's steps, is ahead of PPO, though
not significantly: SAC minus PPO is +3.1 +/- 3.4 points delivered and
+0.096 +/- 0.096 reward (unpaired across training seeds,
`qa/retest/qa_pr9_sac_vs_ppo.log`).

### 2.2 Detours: the earlier pattern does not carry over
Share of delivered packets on a longer-than-shortest path, test seeds:

| Policy | On a detour |
|---|---|
| congestion_aware (best) | 7.6 +/- 1.5% |
| load_aware_routing | 9.2 +/- 1.7% |
| SAC (ties the best) | 15.2 +/- 3.4% |
| PPO | 2.9 +/- 1.4% |
| bandwidth-only rules | 0% (by construction) |

On the random graph, the policies that detoured most delivered least
(`docs/E1_E4_RESULTS.md`, E4). On Abilene, the best rule-based policy
detours on 7.6% of packets and SAC, which matches it, on 15%. The flow bound
is the same with free and with shortest-path routing here too (92.5%), so
detours add no fluid capacity; whatever they contribute is dynamic. The
earlier statement "no case was found where detours help" no longer holds as
a general claim. What the data supports: detours add no capacity in the
fluid model anywhere tested, and whether they help in the simulator depends
on the network.

### 2.3 Where this leaves the overall result
PPO beats every rule-based policy in **two** of four scenarios (bottleneck
+9.7 points, random graph +1.6), and does not on the symmetric grid or on
Abilene. The earlier "three of four" counted the pcap trace scenario, which
was withdrawn with its data.

## 3. Claims, updated

Supported:
* Learned policies beat tuned rule-based baselines on a degraded-node
  scenario and on an irregular graph, not on a symmetric grid and not on the
  real Abilene network with measured traffic, where SAC matches the best rule
  within noise (only just)
  and PPO falls short (while still improving at the end of training).
* Routing matters on Abilene (12.9 points of delivered traffic between
  routing-aware and hop-count rules at the chosen cell, up to 19 across the
  cells tried, on validation seeds); which policies benefit from detours
  depends on the network.

Not supported:
* "RL beats heuristics" without qualification; on the one real-traffic
  scenario it does not.
* "Detours never help."

## 4. Next experiments this suggests
1. **PPO with a longer budget on Abilene** (for example 1M or 2M steps), the
   only scenario where PPO's curve was clearly still rising; a small, cheap
   test of whether 2.1 explains the loss.
2. **Ablations on Abilene** (bandwidth-only and routing-only PPO or SAC):
   routing matters a lot here, so this is a natural place to retest whether
   joint control helps a learner.
3. Other weeks of the Abilene data as held-out traffic (train on one week,
   test on another), which the trace format already supports.

## Changes after QA (18 Oct 2026)
QA findings Q9-2 and Q9-3 (`docs/QA_PR9_REPORT.md`): "SAC does better than
PPO" is replaced by the tested statement (ahead, not significantly), and
"SAC matches the rule" now says the delivered interval only just includes
zero. No numbers changed.

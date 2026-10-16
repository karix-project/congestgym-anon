# Experiments E2 and E3: results

Researcher role, 16 Oct 2026. Plan and rules: `docs/NEXT_EXPERIMENTS.md`,
fixed before any run. Numbers from `qa/research/e2_e3_analysis.py` (`.log`).
Test seeds 2000-2009; agent intervals over 5 training seeds; "wins" are
(training seed, test seed) pairs out of 50. Not yet checked by Tester/QA.

## E2. Tuned PPO on the headline

The configuration was chosen on validation seeds only and the choice was
committed (`1646836`, `results/rl_hp/SELECTED.txt`) before any test result of
the sweep was read: **gamma 0.99, learning rate 3e-4, entropy coefficient
0.01**. It was then trained to 5 seeds and evaluated once on the test seeds.

**Pre-registered comparison:** tuned PPO minus the strongest rule-based
policy (by test reward, among the untuned and E1-tuned ones; that is the
untuned congestion_aware), reward:

> **-0.043 +/- 0.015, tuned PPO ahead in 1 of 50 pairs.**
> **Pre-registered reading: "tuning does not close the gap; on the symmetric
> grid a congestion-aware rule is at least as good as PPO".**

Context (not part of the pre-registered test):

| | Original PPO | Tuned PPO | congestion_aware |
|---|---|---|---|
| Reward | +0.037 | +0.056 | +0.099 |
| Delivered | 73.5% | 73.5% | 74.1% |
| Delay of delivered packets (steps) | 5.82 | 5.58 | 5.16 |
| Deadline misses per delivered packet | 0.370 | 0.357 | 0.330 |

* Tuning made PPO better on exactly the terms it lost on: delay and deadline
  misses, not delivered traffic. It closed about a third of the reward gap
  (from -0.062 to -0.043), not the whole of it.
* Delivered traffic is still a tie with congestion_aware (-0.6 +/- 0.7
  points).
* Against the E1-tuned congestion_aware the gap is -0.037 +/- 0.015: the
  conclusion does not depend on which version of the rule is used.

## E3. Does controlling both halves help the learner?

Rule: joint control helps only where full-control PPO beats **both**
one-half PPOs on reward with the 95% CI above 0.

| Scenario | Full minus bandwidth-only | Full minus routing-only | Joint control helps? |
|---|---|---|---|
| **Bottleneck** | +0.424 +/- 0.042 reward (50/50); +21.9 +/- 1.2 pts | **+0.037 +/- 0.013 reward (49/50); +0.8 +/- 0.4 pts** | **Yes** |
| Random graph | -0.000 +/- 0.006 reward (26/50); +0.5 +/- 0.3 pts | +0.413 +/- 0.011 reward (50/50); +15.0 +/- 0.4 pts | No |
| Headline grid | -0.013 +/- 0.015 reward (19/50); +0.5 +/- 0.6 pts | +0.038 +/- 0.020 reward (46/50); +1.3 +/- 0.6 pts | No |

* **Bottleneck: yes.** Routing carries almost all of the benefit (routing-only
  PPO already delivers 93.8%, against 72.8% with bandwidth only), and
  bandwidth control adds a small but consistent amount on top (+0.037
  reward, ahead in 49 of 50 pairs; +0.8 points of delivered traffic, ahead in
  41 of 50). This is the scenario the plan named as the
  likeliest place.
* **Random graph: no.** Bandwidth carries the benefit; adding routing changes
  the reward by -0.000 +/- 0.006. On delivered traffic the full agent is
  ahead by +0.5 +/- 0.3 points, a small effect on a metric the rule does not
  use; it is reported, not counted.
* **Headline: no** (unchanged from the first matrix).

## What the paper can now claim

Supported:
* A learned policy beats every rule-based baseline, tuned on validation
  seeds, in three of four scenarios (E1), and loses on the symmetric grid
  even after its own hyperparameters are tuned (E2).
* **Which action half matters depends on the network:** routing on the
  bottleneck, bandwidth on the random graph. A learner benefits from
  controlling both only where both carry weight, which here is the
  bottleneck.
* On the symmetric grid, tuning PPO improves delay and deadline misses but
  not enough to match a congestion-aware rule.

Not supported:
* That joint control helps a learner in general.
* That the headline gap is closed by tuning, within the grid searched (gamma,
  learning rate, entropy; network size, batch size and `n_steps` were not
  varied).

## Limits
The E2 grid is small (8 configurations, 3 seeds for selection). E3 used the
original hyperparameters for comparability with the existing full-control
runs. All results are on the 0.4.0 simulator, Python 3.14, Windows, one
machine.

## Changes after QA (16 Oct 2026)
QA finding QF-1 (`docs/QA_E2_E3_REPORT.md`): the bottleneck sentence in E3 no
longer mixes the reward and delivered comparisons. No numbers changed.

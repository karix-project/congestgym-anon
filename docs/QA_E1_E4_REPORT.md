# QA report: experiments E1 and E4

Tester/QA role, 15 Oct 2026. Reviewed: `docs/E1_E4_RESULTS.md`,
`qa/research/e1_*`, `qa/research/e4_detours.*` (uncommitted, branch
`research/next-experiments`). Same-session caveat: not an independent
review.

## Verdict

**Approved, with one wording fix (QE-1).** Every number checks out, and the
tuning was done without touching the test seeds.

## Checks (`qa/retest/qa_e1_e4.py` / `.log`, 44 checks, all PASS)

| # | Check |
|---|---|
| 1 | Selection used validation seeds 1000-1009 only; test seeds 2000-2009 were used once, for exactly one chosen setting per scenario and policy; grid sizes as planned (24 and 4); the default setting is in every grid |
| 2 | **E1's code path is the published evaluation:** the default settings, run through it on test seeds, equal `results/rl/*/qos.csv` exactly (16 episodes, difference 0) |
| 3 | Each chosen setting is the argmax of validation reward, recomputed from the raw episodes; tuned-minus-untuned effects and PPO-minus-strongest figures, recomputed by a different aggregation, equal the doc's |
| 4 | E4 on raw packets: every delivered path is a simple walk on the graph, never shorter than the shortest path; bandwidth-only policies never detour (2,334 to 4,116 packets per case) |
| 5 | Every E4 figure in the doc is in the E4 log |

## Finding

### QE-1 (low): two causal phrasings go beyond the evidence
1. "the routing-aware heuristics detour 6 to 8% of packets, and they are the
   ones that do worse"; and in "What this changes": "detours are what makes
   the routing heuristic worse". The evidence is a correlation.

   One part of it is now measured (`qa/retest/qa_e1_tuned_detours.py` /
   `.log`): on the random graph, tuned congestion_aware detours on
   **2.56 +/- 0.32%** of packets against 7.84 +/- 0.34% untuned (the untuned
   figure reproduces E4 exactly). So "tuning moved congestion_aware towards
   fewer detours" is supported. It is still a correlation, not proof that
   detours cause the loss: the tuning also changed how routing reacts on
   equal-length paths.

   Suggested: "fewer detours go with better delivery: tuning cut
   congestion_aware's detours from 7.8% to 2.6% of packets and raised its
   delivered traffic by 1.8 points", and drop "what makes".
2. "The learner has learned what the fluid bound says" reads as intent.
   Suggested: "PPO almost never detours, consistent with the fluid bound's
   finding that detours add no capacity."

## Not covered
Other tuning grids (the result is "tuned within this grid"); an independent
review in a separate session.

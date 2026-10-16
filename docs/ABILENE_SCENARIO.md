# Abilene trace scenario: choice and a recorded deviation

Researcher role, 16 Oct 2026. Replaces the pcap trace scenario
(`pcap40_grid16_b05`), whose data has no established provenance, with
citable public data (`data/ABILENE_TRACE.md`).

## Decision

**Abilene topology (SNDlib, 12 routers, 15 links), one-week SNDlib trace
(1-7 March 2004) with random episode starts, mean load 120, bandwidth budget
0.7.** Training flags:

```
--topology abilene --nodes 12 --trace data/abilene_trace_20040301_7d.csv.gz
--load 120 --budget 0.7 --trace-random-start
```

Proposed queue name: `abilene_l120_b07`.

## How it was chosen

`qa/research/abilene_select.py` (`.log`) ran 20 cells (load 40 to 120 x
budget 0.5 to 0.8) x 4 heuristics on **validation seeds 1000-1029**, built
exactly as training builds the env. Test seeds were not used. The rules were
fixed in the script before it ran: C1 to C5 as in
`qa/research/headline_settle30.py`, with a fallback of dropping C3 if no cell
qualified.

**Result: no cell met the rules.** Every cell congested enough for C1 failed
C5 (95% CI of the best heuristic's reward at most 0.03): the intervals were
0.046 to 0.085. The fallback did not help, because C3 (routing helps) was
never the problem: routing moves delivered traffic by 8 to 19 points on
Abilene, more than on any synthetic topology tested.

## Deviation from the pre-set rules (approved by the user)

Why C5 failed: with random episode starts over a real week, each test
episode samples a different time of day, and the offered load varies by a
factor of about two over a day. That moves every policy's reward up and down
together. C5's threshold was calibrated on stationary synthetic traffic,
where this does not happen.

The comparisons between policies are what the scenario is for, and they are
stable: the paired 95% CI of (best minus second-best heuristic) reward is
0.019 to 0.031 across the congested cells, with gaps of +0.055 to +0.200
(`qa/research/abilene_select_paired.py`, `.log`).

**Revised rule, for trace scenarios:** C5' = paired CI of (best minus
second-best heuristic) reward at most 0.03, in place of C5. Everything else
as pre-set, including the tie-break (best-heuristic delivered share closest to
76.5%, then lower load).

This rule was written **after** looking at the validation results; it is a
deviation and must be reported as one. It was approved by the user before
any training or any test-seed evaluation of this scenario.

| Load | Budget | Best heuristic delivers | Paired CI (best minus 2nd) | C1-C4 | C5' |
|---|---|---|---|---|---|
| 80 | 0.5 | 87.0% | 0.020 | yes | yes |
| 100 | 0.5 | 73.8% | 0.025 | yes | yes |
| 100 | 0.6 | 81.4% | 0.023 | yes | yes |
| 100 | 0.7 | 85.8% | 0.031 | yes | no |
| 100 | 0.8 | 86.6% | 0.021 | yes | yes |
| 120 | 0.6 | 71.8% | 0.023 | yes | yes |
| **120** | **0.7** | **77.0%** | **0.021** | yes | yes |
| 120 | 0.8 | 78.8% | 0.019 | yes | yes |

Selected: load 120, budget 0.7 (closest to 76.5%). On validation seeds there,
congestion_aware leads, bandwidth control is worth 11.8 points and routing
11.2 points of delivered traffic.

## Consequences
* The training queue replaces `pcap40_grid16_b05` with `abilene_l120_b07`
  (PPO and SAC, 5 seeds each).
* For the paper: report that the trace scenario's stability criterion was
  revised after a first look at validation data, why, and what the
  original rule would have done (no cell qualified).

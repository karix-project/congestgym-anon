# QA report: flow upper bound and RL_RESULTS section 2.6

Tester/QA role, 15 Oct 2026. Reviewed: `carla_gym/bounds.py`,
`scripts/flow_bound.py`, `results/bounds/` (commit `d5e7f57`), and section
2.6 of `docs/RL_RESULTS_0.4.0.md` with `qa/research/lp_static_policy.*`
(uncommitted), in the `feat/rl-scenarios` worktree.

**Independence caveat.** Same session that wrote the code; not a separate
Sonnet session. The checks below use formulations independent of
`bounds.py`.

## Verdict

**Approved, with one wording fix (QB-1) and one addition to the reading
(QB-2).** The LP is correct on every independent check.

## Checks (`qa/retest/qa_bounds.py` / `.log`, all PASS)

| # | Check | Result |
|---|---|---|
| 1 | Analytic bound on the headline with only the total budget binding (fractional knapsack: serve the shortest pairs first) | `bounds.py` 117.961 units/step, knapsack 117.961; difference 7e-14 |
| 2 | A second LP, built from all 140 enumerated shortest paths with path variables | Same value, 117.9606 |
| 3 | pcap expected demand | Sums to the target load, 40.000000 |
| 4 | Free routing equals shortest-path routing | All 35 scenarios, max difference 4e-16 |
| 5 | No policy exceeds the bound except through carried-in backlog | 7 exceedances, all where the bound is 100%, the largest 0.26 points |
| 6 | Section 2.6 figures against `lp_static_policy.log` and `qos.md` | All match (90.3 / 66.8, 100.0 / 62.8, 100.0 / 60.7, 99.4 / 71.1; equal split 68.5) |

The existing unit tests (`tests/test_bounds.py`, 6) cover a shared link, the
budget, free >= shortest, light load and the demand model; they pass.

## Findings

### QB-1 (trivial, wording): "always cheapest" is an explanation, not a result
Section 2.6 explains the free = shortest equality with "spending it on the
shortest routes is always cheapest". The measured fact is the equality in 35
scenarios. In the LP solutions, 1 link (headline) and 2 links (random graph)
sit at allocation 1.0, so per-link limits can bind; the equality still held.
Suggested: "here, spending it on the shortest routes was always at least as
good" or drop "always".

### QB-2 (low, addition): one reason the bound is loose
Check 1 shows the headline LP optimum *is* the fractional knapsack: it serves
the shortest source-destination pairs first and drops the longest pairs
entirely. No policy in this environment can choose which traffic to drop:
admission depends only on buffer space. So the bound credits a capability
that no policy has, which is one concrete reason it sits far above every
policy (90.3% against 74.1%). The reading should say so next to "the bound
is loose".

A fairer reference, under the same budget-only assumption, delivers the
same fraction for every pair: total allocation needed for full delivery is
130.625 units x mean 2.0 hops / 30 = 8.71 against a budget of 7.2, so about
82.7% on the headline. This is not an upper bound on what a policy can
deliver (a policy may favour short pairs indirectly), so it must not be
called one. It is noted only to show how much of the 90.3% comes from the
knapsack effect.

## Not covered
A queueing-aware bound; Python versions other than 3.14; an independent
review in a separate session.

# QA report: PR #5 (simulator 0.4.0)

Tester/QA role, 15 Oct 2026. Reviewed: branch `fix/model-f1-f3` at
`5c54b77`, as pushed to GitHub, from a fresh clone.

**Independence caveat.** This review ran in the same session (Opus 5.5) that
wrote the code, not in a separate Sonnet session as the role plan intends.
To offset that, every check below is new, not a rerun of the PR's own tests.
An independent review in a fresh session is still worth doing before the
paper is submitted.

## Verdict

**Changes requested: one medium finding (Q1), to fix before merge and
before any training run.** Two low documentation findings (Q2, Q3) can go
in the same round. Everything else passed.

## Checks run

Scripts and logs: `qa/retest/qa_pr5.py` / `.log`, `qa/retest/qa_pr5_extra.py`
/ `qa_pr5_extra.log`.

| # | Check | Result |
|---|---|---|
| 0 | Fresh clone of the pushed branch matches the PR head; GitHub reports it mergeable; test suite | PASS: 146 passed, 14 skipped (raw data is not in a clone; those 14 pass in the main checkout), 160 total as claimed |
| A | Invariant fuzzing: 192 episodes (6 topologies x 8 combinations of `queueing`/`tie_break`/`service_order` x bottleneck on/off x 2 seeds), random actions with 15% zeros, load x1.5, no budget. Per step: unit conservation, buffers within capacity, links within allocation, finite reward, each queued packet in exactly one buffer, on its own path, next hop a neighbour, route loop-free | PASS |
| B | Same seed gives identical trajectories for all 8 option combinations, also after an unseeded episode; different seeds differ | PASS |
| C | Offered traffic is identical across all 6 policies on one seed (synthetic and pcap trace), so policy comparisons see the same demand | PASS |
| D | Single-agent and multi-agent envs give equal rewards for the same global action: defaults, legacy options, and the random topology with `topology_seed` 4 | PASS |
| E | Gymnasium `check_env` and PettingZoo `parallel_api_test` with legacy options and with the random topology | PASS |
| F | 180 committed benchmark episodes (grid load x1 / budget 0.5, random load x2 / budget 0.8, pcap load 40 / budget 0.5; 6 policies; 10 seeds) rerun and compared with `results/benchmark/episodes.csv` | PASS: exact match, 0 mismatches |
| G | Legacy options compared with real 0.3.0 numbers from `main` | See Q2 |
| H | Edge cases: grid 4, ring 3, star 2, mesh 2, random 3, grid 64, no budget, zero load, `topology_seed=None`; all-zero, all-one and random actions | PASS |

## Findings

### Q1 (medium): the network's random stream is a copy of the traffic stream
`CarlaGymEnv.reset(seed=s)` seeds both `network.rng` and `traffic.rng` with
`default_rng(s)`, so the two generators start in the same state and produce
the same raw numbers (`qa_pr5_extra.log`: identical state, identical first
five draws). Tie-breaks and service order therefore consume the same bits as
packet counts, sizes and endpoints. Nothing breaks (A to F pass), but the two
sources of randomness are not independent, which is a seeding error a
reviewer checking RNG hygiene will flag. The same holds for the multi-agent
env and for the trace generator.

Before 0.4.0 this did not matter, because the network used its generator
only to build the random graph.

Fix: give the network its own stream derived from the seed, e.g.
`np.random.default_rng([seed, 1])` (or `np.random.SeedSequence(seed).spawn`),
in both `NetworkModel.__init__` and `NetworkModel.reset`, keeping `None` as
"fresh entropy". Add a test that the two generators' states differ after
`reset(seed=s)`. This changes every number again, so rerun the baseline
matrix (about 25 minutes), regenerate the README, rerun the two scenario
sweeps (about 15 minutes) and confirm the headline and second-topology
decisions still hold.

### Q2 (low, docs): "legacy" options do not reproduce 0.3.0 where paths tie
README "Changes in 0.4.0" and the PR say legacy results are "close, not
equal". Measured (delivered fraction, test seeds 2000-2009):

| Scenario | Policy | 0.3.0 | Legacy options on 0.4.0 | Difference |
|---|---|---|---|---|
| grid, load x1, budget 0.8 | max_bandwidth | 0.826 | 0.769 | -0.058 |
| grid, load x1, budget 0.8 | congestion_aware | 0.866 | 0.867 | +0.001 |
| grid, load x1, budget 0.5 | max_bandwidth | 0.519 | 0.494 | -0.025 |
| grid, load x1, budget 0.5 | congestion_aware | 0.540 | 0.536 | -0.004 |
| ring, load x1, budget 0.8 | both | 0.593 / 0.610 | 0.593 / 0.610 | 0.000 |

Exact on the ring (no equal-cost paths), up to 5.8 points off on the grid
with hop-count routing. "Close" understates that. Suggested wording: *the
legacy options reproduce 0.3.0 exactly where shortest paths are unique; where
paths tie, the lowest-id rule differs from networkx's and delivered traffic
can differ by several points (5.8 on the 3x3 grid under equal split).*

### Q3 (low, docs): one range in the PR description
The PR table gives the 0.3.0 spread across source nodes as "0.16-0.30". In
`qa/research/probe_after_fixes.log` the legacy spreads are 0.304, 0.158 and
0.052 (congestion_aware). Write "0.05-0.30 (0.16-0.30 for the hop-count
heuristics)". The other figures checked in the PR, README and commit
messages match their logs: 72-96% corners, 16-20% unused budget, 9-16 edges,
15 edges after, 160 tests.

## Not covered
Python 3.10 to 3.13, Linux and macOS (no CI); the 14 raw-data tests outside
this machine; training under 0.4.0 beyond the smoke test; `render_mode`
`"human"` with a window.

## Round 2 (15 Oct 2026, partial)

Reviewed the **local working tree** of `fix/model-f1-f3`, not GitHub: the
fixes were not yet pushed. Same session, same independence caveat.

| Item | Result |
|---|---|
| Q1 code | **Fixed.** `NetworkModel` draws from `default_rng([seed, 0x6E6574])` at construction and on `reset(seed)`; distinct entropy from the traffic's `default_rng(seed)`, still a pure function of the seed. Two regression tests added; the first fails on the unfixed code |
| Q1 behaviour | **Fixed.** Generators no longer share a state or raw draws (`qa/retest/qa_pr5_round2_extra.log`) |
| A-E, H rerun on the fix | PASS (`qa/retest/qa_pr5_round2.log`; `qa_pr5.py` was run cut off before check F, since the benchmark was still being regenerated) |
| Q2 | Wording fixed in substance, but the sentence before it now lacks a full stop: `...service_order="id"` runs into `It reproduces...` (R2-1) |
| Q3 | Not yet done (PR description still to be edited) |
| F, benchmark reproduction | Still to do, against the regenerated `results/benchmark/` after the push |

### R2-1 (trivial, docs)
README "Changes in 0.4.0": add a full stop after
`service_order="id"`.

### R2-2 (low, pre-existing): the `random` baseline draws from the traffic stream
Same class of error as Q1, older than this PR. `scripts/run_benchmark.py`
(line 86), `scripts/evaluate_saved.py` (line 83) and
`examples/random_agent_test.py` (line 42) build the policy generator with
`default_rng(seed)` for the same `seed` as the episode's traffic, so the
`random` baseline's actions are the traffic's own random numbers. Only the
`random` rows are affected; the heuristics are deterministic and the RL
agent does not use this generator. Fix as for Q1 (a tagged seed). Fixing it
changes the `random` rows of the benchmark, so either fix it before the
benchmark is regenerated for this PR, or record it as a known issue.

### Round 2 verdict
Q1 is fixed and verified. Approval waits on: R2-1, Q3, a decision on R2-2,
and check F against the pushed, regenerated results.

## Round 3 (15 Oct 2026): approved

Reviewed the pushed branch at `859ba4d` from a fresh clone. Same session,
same independence caveat.

| Item | Result |
|---|---|
| Tests | 149 passed, 14 skipped in the clone (raw data); 163 in total, as claimed |
| A-E, G (`qa/retest/qa_pr5_round3.log`) | PASS. Legacy gap to 0.3.0 unchanged (legacy mode draws no random numbers) |
| F: 180 committed benchmark episodes rerun | PASS, exact. Includes the `random` rows, so the regenerated benchmark was produced with the R2-2 fix |
| Q1 (`qa_pr5_round3_extra.log`) | Fixed: network and traffic generators share no state and no draws |
| R2-1, R2-2, Q3 | Fixed |
| Figures in README, docs and PR description against their source files (`qa_pr5_round3_audit.py` / `.log`, 16 checks) | PASS: trace drop ranges, the "about 95%" claim, the code-review table and speeds, spreads, test count, and the 30-seed headline selection, which reproduces from its summary file |

**Verdict: approved.** No open findings. Still not covered: Python 3.10 to
3.13, Linux and macOS (no CI), and an independent review in a separate
session.

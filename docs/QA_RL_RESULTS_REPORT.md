# QA report: 0.4.0 training results and their first reading

Tester/QA role, 15 Oct 2026. Reviewed: `results/rl/` (medium matrix, 55
runs), `scripts/evaluate_saved.py` as extended, and
`docs/RL_RESULTS_0.4.0.md`, all uncommitted in the `feat/rl-scenarios`
worktree.

**Independence caveat.** Same session (Opus 5.5) that produced the runs and
the reading; not a separate Sonnet session. The checks use code paths
separate from the Researcher's analysis where possible.

## Verdict

**Approved, with two wording fixes to the reading (QR-1, QR-2).** No
problem found in the runs, the evaluation or the numbers.

## Checks (`qa/retest/qa_rl_results.py` / `.log`, all PASS)

| # | Check |
|---|---|
| 1 | 55 run directories (35 PPO, 20 SAC), each with its five output files; no Traceback, Error or nan in any `train.log`; every run trained to its full budget (PPO 500k, SAC 100k) |
| 1 | Every saved model has the observation and action size of its scenario and ablation (e.g. random graph 39 / 30, routing-only ablations half the action), so topology and `--control` were applied as intended |
| 2 | The headline env is the swept scenario: congestion_aware delivers 74.1% on test seeds against 73.7% in the 30-seed validation sweep; queue flags match the decision |
| 3 | Training (0-7), validation (1000-1004) and test (2000-2009) seeds are disjoint; the QoS tables use test seeds only |
| 4 | PPO minus best heuristic reward, recomputed from `final_eval.csv` instead of `qos.csv`, matches the write-up in all four scenarios (-0.062, +0.173, +0.056, +0.025; wins 0, 50, 50, 44 of 50); bootstrap intervals agree with the t-intervals and exclude zero in every case |
| 5 | Every figure quoted in `RL_RESULTS_0.4.0.md` is backed by `rl_results_analysis.log` or a `qos.md` table, including the 0.3.0 comparison figure from `PUBLICATION_READINESS.md` |
| - | `evaluate_saved.py` self-check: agent and deterministic-baseline rewards equal `final_eval.csv` to 2.2e-16 in all seven folders; 10 new tests; 177 tests in the worktree (163 pass, 14 skip for want of raw data) |

## Findings

### QR-1 (trivial): contradictory convergence wording
Section 2.3 lists pcap SAC under "Flat" (-0.003 +/- 0.025), and in the next
bullet calls SAC intervals of +/- 0.025 to 0.060 "too wide to call". Move
pcap SAC to the SAC bullet.

### QR-2 (low): a mechanism stated as a finding
Section 2.1: "PPO matches the throughput but serves traffic in a worse order
or over longer paths." Delay and deadline misses were measured; order and
path length were not. Either say "possibly because ...", or measure mean
hop count of delivered packets for PPO and congestion_aware.

### Notes, not defects
* The best heuristic is picked per scenario and per metric on the test seeds.
  That favours the heuristics, so it makes the RL wins conservative, not
  inflated. Worth one sentence in the paper.
* 5 training seeds is the minimum the field accepts. The effects claimed are
  large relative to their intervals (bootstrap and t agree), so the
  conclusions do not hinge on it.
* One hyperparameter set, untuned. The reading already says so.

## Not covered
Results under other hyperparameters or step budgets; Python versions other
than 3.14; an independent review in a separate session.

# QA report: PR #9 (SNDlib Abilene data and scenario)

Tester/QA role, 18 Oct 2026. Reviewed: branch `data/sndlib-abilene` at
`21ff92e`, from a fresh clone of GitHub. Same-session caveat: not an
independent review.

## Verdict

**Approved, with two low findings (Q9-2, Q9-3) and one process note (Q9-1).**
The bug is fixed and guarded, the corrected selection was recorded before its
results existed, the data reproduces from SNDlib's archive, and every
reported figure recomputes.

## Checks (`qa/retest/qa_pr9.py` / `.log`, 34 checks, all PASS)

| # | Check |
|---|---|
| 0 | Fresh clone: 198 passed, 12 skipped (raw data and the SNDlib archive are not in a clone); 210 in total |
| 1 | The bug is gone: every Abilene episode (training evaluation and benchmark) has positive delay (minimum 4.16 steps) and counted deadline misses; the discarded buggy run is not in the repository; **the two regression tests fail when the buggy generator from `c3cd097` is put back** |
| 2 | Selection integrity: the corrected choice (`4accaca`) was committed before the Abilene results; the rule, recomputed from the raw validation episodes, qualifies (80, 0.5) and (120, 0.8) and picks (120, 0.8); validation seeds only; the buggy run's logs are kept |
| 3 | Data: no pcap or cs448b trace and no reference to them in code, notebooks or README; the committed trace equals `prepare_sndlib_trace.py`'s output from SNDlib's archive; `data/ABILENE_TRACE.md` carries the citations and the licence caveat |
| 4 | PPO and SAC minus congestion_aware, recomputed from `final_eval.csv`: -0.133 (3/50) and -0.037 (16/50), as reported; every figure in `docs/ABILENE_RESULTS.md` is backed by `abilene_results.log` |
| 5 | Four committed Abilene benchmark episodes reproduce; the README says "two of the four scenarios" and its four drop ranges match the table; notebooks 02-04 are executed with no errors |

## Findings

### Q9-1 (process): PR #9 has no CI yet
The GitHub Actions workflow is added by PR #8, so PR #9 shows no checks.
Merge PR #8 first, then update PR #9 from `main` so CI runs on it before it
is merged.

### Q9-2 (low): "SAC does better than PPO" is not established
`docs/ABILENE_RESULTS.md` 2.1: "SAC, at a fifth of PPO's steps, does
better." The runs are independent, so the comparison is unpaired across
training seeds (`qa/retest/qa_pr9_sac_vs_ppo.py` / `.log`): SAC minus PPO is
**+3.1 +/- 3.4 points** delivered (includes zero) and **+0.096 +/- 0.096**
reward (lower bound at zero). Suggested: "SAC, at a fifth of PPO's steps, is
ahead of PPO, though not significantly."

### Q9-3 (low): "SAC matches the rule" is borderline
SAC minus congestion_aware, delivered: -2.3 +/- 2.3 points, an interval
whose upper end is +0.03. "Matches within noise" is correct but should say
the interval only just includes zero.

## Not covered
CI on Linux and other Python versions (Q9-1); an independent review.

# QA report: README with the 0.4.0 results

Tester/QA role, 15 Oct 2026. Reviewed: `README.md` on `feat/rl-scenarios`
at `b8d85ba`, from a fresh clone of GitHub. Same-session caveat as the
earlier reports: not an independent review.

## Verdict

**Approved. No findings.** Two checks flagged on first run were artefacts of
the checks themselves, explained below.

## Checks (`qa/retest/qa_readme_040.py`, log `qa_readme_040.log`)

| # | Check | Result |
|---|---|---|
| 1 | Test suite in the clone | 169 passed, 14 skipped (raw data is not in a clone); 183 with raw data, as the README states |
| 2 | Regenerate every table from the results files (`rl_overview.py`, `flow_bound.py`, `update_readme_results.py`) | **README content unchanged**: `git diff` is empty. Git listed the file as modified only for line endings (the script writes LF; the Windows checkout is CRLF) |
| 3 | Every file path the README names exists | 18 of 19 exist; the 19th is `data/raw/`, which the README itself describes as git-ignored and local-only, so its absence from a clone is correct |
| 3 | Documented commands run | The training command lists the 55 runs reported; `evaluate_saved.py` runs on the headline folder and passes its self-check |
| 4 | Prose claims against the tables | "3 of 4 scenarios", "most on the bottleneck" (+9.6 against +3.9 and +1.6 points), the headline description (delivered 73.5 vs 74.1, delay 5.82 vs 5.16, misses 0.37 vs 0.33, reward +0.037 vs +0.099), "both halves do not help PPO" (-0.013 +/- 0.015), free = shortest bound in all 35 scenarios, test count |
| 5 | Packaging | `bounds` extra in `pyproject.toml` and in `all`; SciPy in `requirements.txt` and pinned in `constraints.txt` |

## Not covered
Installing from a fresh virtual environment (`pip install -e ".[all]"`),
Python versions other than 3.14, Linux and macOS (no CI).

# Developer resume notes: continuing the publication-ready work

Written by the Developer role, 11 Oct 2026, for the Developer role at a later
date. The user postponed the publication work to focus on the university
presentation. **Nothing is running. No training was launched.**

> **Update, 15 Oct 2026 (Developer).** Sections 1 and 5 below are out of
> date. The simulator is now 0.4.0 (`docs/CODE_REVIEW_FOR_PUBLICATION.md`,
> fixes F1-F3), merged into `feat/rl-scenarios`. The training queue was
> rebuilt for it from `docs/HEADLINE_SCENARIO.md`: scenarios
> `grid_l125_b06` (headline), `grid_bottleneck_b08`, `pcap40_grid16_b05`,
> `random9_l15_b06`; PPO ablations `grid_l125_b06_bandwidth_only`,
> `grid_l125_b06_routing_only`, `random9_l15_b06_routing_only`. 55 runs at the
> default budgets (`--list` shows them). `grid_b08`, `ring_b08` and
> `grid_load2_b08` were dropped. All 11 configurations passed a 2,500-step
> smoke test on 15 Oct. The run size still has to be agreed with the user
> before launch, and `scripts/evaluate_saved.py` still does not know
> `--bottleneck`, `--trace`, `--control` or `--topology` beyond its own flags
> (section 6, item 2).

Read this file, then `docs/PUBLICATION_READINESS.md` (what a venue would
reject and why) and `docs/SLIDES_ABSTRACT_REVIEW.md` section 5 (the plan).

## 1. State of the repository

| Thing | State |
|---|---|
| `main` (local and GitHub) | `79e2150`. Coding part complete, tested, confirmed. 129 tests pass (115 passed, 14 skipped in a fresh clone). |
| `research/publication-readiness` (local only) | `main` + 2 doc commits: `docs/PUBLICATION_READINESS.md`, `docs/SLIDES_ABSTRACT_REVIEW.md` |
| `feat/rl-scenarios` (local only, **current branch**) | the above + training tooling (`e2fa918`) + the two handoff reports. 133 tests pass. |
| Unpushed work | Everything on the two local branches. **Not on GitHub.** If this machine is lost, it is lost. Ask the user whether to push. |
| Pull requests merged | #1 (QA round), #2 (final researcher pass, trace recalibration), #3 (QA record) |
| Running processes | none |

First commands in a new session:

```bash
git status -sb
```
```bash
git log --oneline main..HEAD
```

## 2. Environment facts (they cost time to rediscover)

* Repo: `C:\Users\aryan\Downloads\carla-gym\carla-gym`. The venv path in the
  original handoff does not exist.
* `python` is **not on PATH**. Use `.venv\Scripts\python.exe` (Python 3.14.5).
  Base interpreter: `C:\Users\aryan\AppData\Local\Programs\Python\Python314\python.exe`.
* `gh` is not on PATH: `C:\Program Files\GitHub CLI\gh.exe`, signed in as
  `spaciouskarter78`, admin on the repo. The repo has **no CI**.
* 12 logical cores. Measured: PPO about 900 steps/s alone; ten runs in
  parallel (5 PPO x 500k + 5 SAC x 100k) took about 50 minutes. SAC is about
  ten times slower per step than PPO.
* Raw datasets live only in git-ignored `data/raw/` on this machine. Tests
  that need them are skipped elsewhere.
* `pypdf` was added to `.venv` to read the user's PDFs. It is not in
  `requirements.txt` and is not needed by the project.
* Tooling traps met in the first session:
  * The Write tool refuses to overwrite a file changed on disk since it was
    read; patch with a small Python script instead.
  * Heredocs containing triple quotes broke Bash parsing; write patch scripts
    to the scratchpad and run them.
  * `git reset` after `git rm --cached` silently undid the untracking (bug T1).
    Check `git ls-files` after any index surgery.
  * A QA log that printed a diff re-introduced a removed sensitive string.
    Scan logs before committing them.
  * Foreground `sleep` is blocked; wait on long jobs with a PowerShell loop
    that polls for an output file.

## 3. Standing rules from the user

* Five roles; state the current role and ask which role is wanted at the end
  of every reply. Tester/QA and Editor/Presenter are meant to run on Sonnet;
  say so if the session model differs.
* Warn before steps that could hit token or context limits. Long jobs run in
  the background with logs in files.
* No fabricated numbers; every figure must come from a run and a file.
* Do not push, merge, delete branches or rewrite history without an explicit
  instruction for that action.
* Decisions already made: raw datasets removed from current files; **git
  history deliberately not purged**; pcap trace documented at load 40; cs448b
  is a stress case.
* Decisions still open: project rename (CARLA name clash), email in `LICENSE`,
  target venue, whether to replace the traces with public citable data.

## 4. What is ready to use

Added on `feat/rl-scenarios`, tested, not yet in the README:

* `carla_gym/wrappers.py`: `ControlSubsetWrapper(env, "bandwidth" | "routing")`
  for agents that control one half of the action.
* `examples/train_agent.py`: `--control {both,bandwidth,routing}` and
  `--bottleneck 4:0.5`.
* `scripts/run_training_matrix.py`: queues the runs, N at a time, skips
  finished ones (a run is finished when its `final_eval.csv` exists), safe to
  restart. Output goes to `results/rl/<scenario>/<algo>_seed<N>/`.

All six configurations passed a 2,500-step smoke test on 11 Oct.

Show the queue without running anything:

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/run_training_matrix.py --list
```

## 5. The postponed step: launching the training runs

The user approved the compute, then stopped on seeing the 8-hour estimate and
dismissed the size question. **Ask again before launching.** Options offered:

| Option | Flags | Runs | Rough time |
|---|---|---|---|
| Full (recommended) | defaults: PPO 1M, SAC 200k, 5 seeds | 60 | about 8 h |
| Medium | `--ppo-steps 500000 --sac-steps 100000` | 60 (or reuse the old grid results) | about 4 h |
| PPO only | `--no-sac --ppo-steps 500000` | 35 | about 2 h |
| Lean | `--no-sac --ppo-steps 500000 --seeds 3` | 21 | about 1.5 h |

Scenarios in the queue: `grid_b08`, `ring_b08`, `grid_bottleneck_b08`
(node 4 at 0.5), `grid_load2_b08`, `pcap40_grid16_b05`; ablations
`grid_b08_bandwidth_only` and `grid_b08_routing_only` (PPO only).

Launch it **detached** so it survives the chat, from PowerShell:

```powershell
$env:PYTHONPATH = "."; Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "scripts\run_training_matrix.py","--workers","10" -RedirectStandardOutput "results\rl_matrix.log" -RedirectStandardError "results\rl_matrix.err"
```

Then: ask the app to keep the machine awake, tell the user to keep it plugged
in, and watch `results\rl_matrix.log` (one line per finished run, "ALL DONE"
at the end). Each run also writes `train.log` in its own folder.

Why the budgets are unequal: PPO's curve was still rising at 500k
(+0.444, +0.448, +0.460 at the last three checkpoints); SAC was flat by
60k to 100k. Equal step counts would cost about 60 CPU-hours. The paper
should compare *converged* values and report sample efficiency separately.

Be ready for RL **not** beating the heuristics in some scenario (the
double-load grid is deep in overload; ring has little routing freedom). Report
it as it comes out.

## 6. After the runs finish

1. `scripts/summarize_training.py results/rl/<scenario>` for each scenario
   (table and learning-curve figure). The ablation folders hold PPO only, so
   the script works on them as they are.
2. `scripts/evaluate_saved.py` for delivered and dropped tables. **It does
   not yet know about** `--bottleneck`, `--trace` / `--load`, or `--control`:
   it rebuilds the env from `--topology/--nodes/--budget/--load-scale` only.
   Extend it (reuse `make_env` from `train_agent.py`) before using it on the
   bottleneck, trace and ablation runs.
3. Decide what happens to the old `results/grid_b08/` (PPO 500k, SAC 100k).
   Cleanest: the README headline moves to `results/rl/grid_b08/` and the old
   folder is removed in the same commit, with the README regenerated by
   `scripts/update_readme_results.py` (its `SOURCES` paths must be updated).
4. Update notebook 04 (`RESULTS = "results/grid_b08"`) and the Researcher's
   scripts in `qa/research/` (they read `results/grid_b08`).
5. README: add `wrappers.py`, `run_training_matrix.py`, `--control`,
   `--bottleneck` to the layout and usage; add a per-scenario RL results
   table and an ablation table.
6. Hand to Tester/QA, then Researcher.

## 7. The rest of the publication plan

From `docs/PUBLICATION_READINESS.md` section 5. None is started.

| # | Work | Notes for implementation |
|---|---|---|
| 1 | RL on more scenarios | Section 5 above |
| 2 | Trained ablations | In the same queue |
| 3 | ECMP and inverse-capacity baselines | Add to `carla_gym/baselines.py`. The simulator routes each packet on one Dijkstra path, so true ECMP needs either per-packet random tie-breaking among equal-cost paths or a small simulator change; decide and document. With uniform capacities, inverse-capacity weights equal hop count, so that baseline only differs in bottleneck scenarios. |
| 3b | Multi-commodity-flow upper bound | Linear program (`scipy.optimize.linprog`; add SciPy as an optional extra). Bound the delivered fraction per scenario from the expected traffic matrix and link capacities x budget. State clearly that it is a fluid bound, not achievable by the slotted simulator. |
| 4 | Scale: 25, 36, 49 nodes | `NetworkModel.update_routing` runs all-pairs Dijkstra **every step**; this will dominate at 49 nodes. Measure steps/s first. Caching when weights are unchanged is an easy win. |
| 5 | Volume-weighted reward | A reward option weighting each step by its offered load. Changes the objective, so everything retrains. Treat as future work unless the user asks. |
| 6 | Citable data, rename, CI | Data: SNDlib (Abilene, GEANT traffic matrices) or MAWI; check terms first; a download needs the user's permission. Rename touches the package name, `CarlaGym-v0`, README, notebooks, docs. CI: GitHub Actions matrix, Python 3.10 to 3.13 on ubuntu and windows, `pip install -e ".[all]"`, `pytest -q`. The code has only been *parsed* under the 3.10 grammar, never run; expect small fixes. |
| 7 | Related-work table | Researcher's job. Papers to read in full: PackeRL (arXiv 2410.10377), PRISMA (IFIP Networking 2022), ns3-gym, NetworkGym, mobile-env, Park, NetSimGym. |
| 8 | Validate against ns-3 | One to two weeks. Needed for research journals only. |

Minimum for a workshop, student track or indexed conference: 1, 2, 3, 6, 7.
Add 4 for a software journal (JOSS or SoftwareX, the best fit).

## 8. Known loose ends

* `docs/` reports refer to "35 scenarios" and load 110 in older sections;
  they are dated records and were left as written.
* README statement "RL results are reported for one training scenario" must
  change once section 5 is done.
* `data/raw/`, the old notebook 03 output and one MAC-vendor string are still
  in the public git history by the user's decision.
* Untested: Python 3.10 to 3.13, Linux, macOS, `render_mode="human"` with a
  real window, multi-agent training, TD3 beyond a smoke run.
* AI assistance must be disclosed to whatever venue is chosen.

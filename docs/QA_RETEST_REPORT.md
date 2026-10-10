# QA re-test report: branch `fix/qa-round-1`

Tester/QA role, 10 Oct 2026. Re-test of commit `1bc5f4e` against
`docs/TESTER_REPORT.md`. Windows 11, Python 3.14.5, package versions as in
`constraints.txt`. Run on Opus 5.5 (the project rules name Sonnet for this
role). Nothing was fixed in this pass; fixes belong to the Developer role.

Scripts and logs: `qa/retest/` (`retest_probes.py`, `probes.log`,
`repro_compare.log`, `pytest.log`, `fresh_clone.log`, `ppo_seed0_rerun.log`).

## Verdict

The Developer's fixes hold up and the reported results are reproducible from
the files in the repo. Two defects were found (T1, T2); both are now fixed and confirmed (section 8).
Eight smaller items (T3 to T10) remain for a fix-or-document decision.

## 1. Defects to fix (send to Developer)

### T1 (MEDIUM): `__pycache__` and `.idea` are still tracked
`git ls-files` still lists 14 files: 6 under `.idea/` and 8 `.pyc` files under
`carla_gym/**/__pycache__/`. Commit `a660a6c` says it untracked them and
`docs/TESTER_REPORT.md` (M6 row) repeats that, but the commit does not touch
those paths. `.gitignore` does not help for files already tracked. M6 is
therefore only partly fixed, and the tester report contains an incorrect
statement. Fix: `git rm -r --cached` on those paths, and correct the report.

### T2 (MEDIUM): non-finite actions bypass link capacity and the budget
An action containing NaN is accepted. NaN survives `np.clip`, and every
comparison against a NaN link budget is false, so the link is treated as
having unlimited bandwidth.

| Constant action | Mean reward | Max units served on one link per step (capacity 30) |
|---|---|---|
| all NaN | **+0.768** | **79.9** |
| all ones | +0.328 | 24.0 |
| all zeros | -1.587 | 0.0 |

+0.768 is higher than the trained PPO agent (+0.494). No reported result is
affected (the saved models reproduce their rewards with finite actions, and
the fuzz run found no capacity violation with finite actions), but the
environment must not reward an invalid action. Fix: reject non-finite actions
in `CarlaGymEnv.step` and `MultiAgentCarlaGymEnv.step` with a `ValueError`,
and add a test.

## 2. Smaller items (decide: fix or document)

| ID | Item | Evidence |
|---|---|---|
| T3 | `delay_reference=0` is accepted by the constructor and raises `ZeroDivisionError` at the first step. Negative `k`, `c`, `m`, `load_scale` are accepted silently. | probe "Invalid constructor arguments" |
| T4 | Multi-agent `step()` accepts an action dict with an agent missing; that agent's links silently get bandwidth 0 and routing weight 1. Not documented. | probe "MARL: an agent missing" |
| T5 | Deadline-miss denominator counts a packet twice if it expires in a queue and is later delivered (once at expiry, once at delivery); the numerator counts it once. The README says each packet is counted once, which is true of misses only. Expiry is flagged one step after the miss became certain. Totals over an episode are unaffected. | probe "M1 deadline boundary" |
| T6 | No backpressure: a packet fully transmitted to a neighbour whose buffer was full at the start of the step is dropped, even if that neighbour emptied during the step. Two full neighbours sending to each other lose a packet. This follows the documented start-of-step rule, but the README does not spell out the consequence. | manual probe, section 4 |
| T7 | After a non-looping trace ends, offered load is 0, so the throughput term is 0 while the backlog drains (197 units delivered for no throughput credit in the probe). Rewards stay finite. | probe "loop=False" |
| T8 | Wording left from the development phases: "Sprint" (9 places in `examples/random_agent_test.py` and `tests/test_env.py`), "Phase-1", "Phase 2". Harmless, but not publication wording. | probe "Repo text" |
| T9 | The observation's pending-demand entry counts a part-transmitted packet at full size. Slight overstatement, clipped to 1. | code review |
| T10 | The `random` baseline row in the results table comes from training-seed 0's run; its value differs slightly between run directories (-0.889 to -0.908) because its action RNG is seeded per run. The deterministic baselines are identical in all ten directories. | probe "README / report numbers" |

## 3. Re-check list from `docs/TESTER_REPORT.md` section 9

| # | Item | Result |
|---|---|---|
| 1 | `pytest` | **98 passed**, here and in a fresh clone |
| 2 | Repro scripts vs `qa/after` logs | all 7 identical |
| 3 | H1 with my own script through SB3 | PASS: 4 envs x 4 auto-reset episodes, 16 traffic sequences, no two identical; same constructor seed gives the same episode in 3 separate processes |
| 4 | `forward()` line-by-line review | No defect with finite inputs. Notes T5, T6, T9. A dropped packet's link budget is consumed (confirmed: dropped 10.0, link served 10.0). A delivery into a full destination node succeeds, which is correct for a sink. |
| 5 | `pkt.sent` never carries to another link | PASS: checked on every queued packet over 40 fuzz runs x 150 steps; also "part-sent packet is at the head of its queue", "at most one hop per step", "sent never decreases without moving" |
| 6 | Deadline boundary `age > deadline` | Acceptable. Delivery with delay equal to the deadline is not a miss (checked). See T5. |
| 7 | Deadline-miss denominator noise | At load x0.1 (1.8 packets resolved per step): term mean 0.004, std 0.043, zero on 99% of steps. Not a practical problem at the loads used. |
| 8 | Changed sensitivity check in `random_agent_test.py` | Agreed. Routing weights have no monotone effect on reward, so the bandwidth-only correlation is the right check. All six checks pass. |
| 9 | `step()` after truncation with wrappers | PASS: `gym.make` wrappers, `RecordEpisodeStatistics`, SB3 `check_env` (native and rescaled env), `DummyVecEnv` auto-reset |
| 10 | Multi-agent ownership on other seeds and sizes | PASS: 166 envs (random topology with 5 sizes x 30 seeds, plus fixed topologies): every link owned once by an incident node, no empty agent, one step runs |
| 11 | Several bottlenecks; star hub | PASS: min-scale rule on shared links, hub scales all 8 links, scale 1.0 is a no-op, reward decreases with scale, multi-agent env accepts the option |
| 12 | `loop=False` past the end of a trace | PASS (finite, no crash). See T7. |
| 13 | Training reproducibility | Close, not bit-exact. PPO seed 0 re-run for the full 500,000 steps: test reward +0.492 against the saved +0.496; learning curve differs by 0.016 on average (0.045 at most) over 20 checkpoints. The re-run used 2 CPU threads, the original 1, which is the likely cause. The difference is well inside the reported +/- 0.013 interval. The deterministic baselines are identical. |
| 14 | README commands from a fresh clone | PASS: clone, new venv, `pip install -r requirements.txt -c constraints.txt`, 98 passed, `pip install -e ".[all]"`, import from outside the repo; working tree clean afterwards. Also ran: training CLI on the pcap trace, TD3, the `train_ppo.py` shortcut, a notebook without `PYTHONPATH`. |
| 15 | Notebook 04 text vs plots | PASS: printed statements are computed from the same data as the plots; all four notebooks are executed in order with 0 errors and no stale references |

## 4. Additional checks

* **Simulator invariants under fuzz.** 5 topologies x 8 seeds x 4 action
  patterns (uniform random, sparse, tiny allocations with changing routes,
  on/off bursts), with budgets None / 0.5 / 0.8 / 1.0, loads x0.4 to x2.5, and
  two bottleneck nodes. Checked every step: conservation, buffer bound,
  per-link service within allocation, budget, observation bounds, each reward
  term in [0, 1]. No violation. Reward range seen: [-2.000, +0.760], inside
  the documented [-2, 1].
* **Edge configurations.** 2-node ring and star, 12-node mesh, 25-node grid,
  20-node random, zero load, load x8: every baseline runs on each.
* **Reported numbers.** Recomputed from `final_eval.csv` and `episodes.csv`
  with my own code: all 8 rows of the training table and all 210 cells of the
  scenario matrix match the README; confidence intervals match.
* **Determinism.** A benchmark scenario recomputed on the current code equals
  the saved values (max difference 1e-15). Saved PPO and SAC models reload and
  reproduce their saved test rewards (max difference 6e-17).
* **Documented facts.** Mean offered load over 2,000 steps is 102.8 (README:
  "about 104"); E = 12, N = 9, action (24,), observation (33,).
* **Datasets.** Both committed traces: schema, node range, no self-loops, no
  duplicate (step, src, dst), positive volumes.

## 5. Requirements check against the abstract

| Requirement | Met? |
|---|---|
| Python RL framework on the Gymnasium API | Yes |
| Dynamic packet routing | Yes |
| Bandwidth allocation | Yes |
| Node-bottleneck mitigation | Yes (explicit scenario and option) |
| Customizable topologies | Yes |
| Volatile data-load generation | Yes |
| Granular QoS reward (packet loss, latency) | Yes |
| Train, evaluate, benchmark RL agents | Yes |
| RL agents vs rule-based traffic-engineering baselines across diverse scenarios | Partly: baselines are benchmarked across 35 scenarios, RL agents in one scenario only |
| PPO and DQN (named in the slides) | PPO yes; DQN no (SAC used instead, because the action space is continuous). For the Researcher. |

## 6. Not verified in this pass

* Python 3.10 to 3.13, Linux, macOS (not available on this machine).
* A real window for `render_mode="human"`.
* Multi-agent training.
* Scientific soundness of the model and reward: that is the Researcher's pass.

## 7. Recommended next steps

1. **Developer:** fix T1 and T2; decide T3 to T10 (fix or document); correct
   the M6 statement in `docs/TESTER_REPORT.md`.
2. **Tester/QA:** short confirmation pass on those changes.
3. **Researcher.**

## 8. Confirmation pass on T1 and T2 (commits `b6cc00e`, `29b9836`)

Log: `qa/retest/confirm_t1_t2.log`, `qa/retest/probes_round2.log`.

| ID | Check | Result |
|---|---|---|
| T1 | Files matching `__pycache__`, `.idea`, `*.pyc`, `.DS_Store` in `HEAD` | 0 |
| T1 | Same search in a fresh clone of the branch | 0 |
| T1 | The 8 `.pyc` files that exist on disk after running the tests | ignored by git, not shown as changes |
| T2 | 90 actions with one NaN / +inf / -inf entry at a random position, 5 topologies | 90 rejected, 0 accepted |
| T2 | NaN through `RescaleAction`, `gym.make`, SB3 `DummyVecEnv`, a Python list, the multi-agent env | all rejected with `ValueError` |
| T2 | Finite out-of-range actions (1e6, -5) | still accepted and clipped; most a link served is 30.0 of capacity 30 |
| T2 | Env state after a rejected step | untouched: the following 20 rewards equal those of an env that never saw the bad action |
| Regression | `pytest` | **114 passed** |
| Regression | My probe script re-run | same outcome as before minus the NaN note: saved benchmark values, saved models and README numbers all still reproduce |

**T1 and T2 are confirmed fixed.** The one "FAIL" line in the probe log is my
own probe's wrong expectation (its two packets share a link); the behaviour
is the documented start-of-step rule, recorded as T6.

Still open: T3 to T10 (section 2), none of which blocks correctness of the
reported results. Whether to fix or document them is a Developer/user call.

## 9. Confirmation pass on T3 to T10 (commits `b5d2cee`, `129ab50`)

Script and log: `qa/retest/confirm_t3_t10.py`, `qa/retest/confirm_t3_t10.log`.
26 checks, 0 problems.

| ID | Developer's decision | Confirmed? |
|---|---|---|
| T3 | Fixed | Yes. Six invalid parameter sets are rejected at construction by both envs; zero values and a tiny `delay_reference` still run; explicit defaults behave exactly like implicit ones. |
| T4 | Fixed | Yes. Missing, extra and misnamed agents are rejected on all five topologies; a full action dict is accepted afterwards and the episode ends cleanly. |
| T5 | Documented | Yes, in `utils/reward.py` and the README limitations. One wording residue: the README reward section still ends the DeadlineMiss bullet with "Each packet is counted once", which is true of misses but reads oddly next to the limitation about the denominator. Cosmetic. |
| T6 | Documented | Yes. |
| T7 | Documented | Yes. |
| T8 | Fixed | Yes. No "Sprint"/"Phase" or informal wording in tracked code, README or notebooks; the sanity script prints "Sanity checks" and passes 6/6. |
| T9 | Documented | Yes. |
| T10 | Fixed | Yes. Pooled baselines recomputed from the ten raw `final_eval.csv` files equal `summary.csv` and the README; only `random` moved (-0.908 to -0.901); PPO and SAC rows unchanged; notebook 04 shows the pooled value with 0 errors. |

I agree with documenting T5, T7 and T9 rather than changing them: the code
diff for rounds 2 and 3 touches only validation and docstrings
(`traffic.py` +2 lines, `marl_env.py` +5, `reward.py` +14 of which 4 are
checks), so the trained models and tables remain valid, and the benchmark
determinism and saved-model probes still reproduce to 1e-15.

Regression: **122 passed** here and in a fresh clone (working tree clean
afterwards).

Note on `qa/retest/probes_round3.log`: it shows two "FAIL" lines. Neither is
a defect. One is my mis-specified probe from the first pass (T6). The other is
the "agent missing from the action dict" probe, which expected the old silent
acceptance and now hits the T4 `ValueError`, as intended.

### Tester/QA status

No open defects from this role. Remaining before submission, none of which
is a code-correctness issue:

* the user's four decisions (raw datasets with real IP addresses in the
  public repository, the email in `LICENSE`, the project name, push / PR);
* environments this machine cannot test (Python 3.10 to 3.13, Linux, macOS);
* the Researcher pass.

## 10. Confirmation pass on round 4 (commits `7901822`, `b9d3857`)

Round 4 addressed the three inconsistencies in `docs/RESEARCHER_REVIEW.md`
section 4. Script and log: `qa/retest/confirm_round4.py`,
`qa/retest/confirm_round4.log`. 17 checks.

| # | Item | Confirmed? |
|---|---|---|
| 1 | Episode-aggregate delivered / dropped / delay / deadline figures | Yes. I recomputed them without the new helper (delays captured directly from `NetworkModel.forward`, totals from raw counters) for 36 benchmark episodes across 6 scenarios including both traces and the bottleneck, and for a PPO and a SAC agent: agreement to 1e-15. All 210 benchmark cells and all 8 trained-agent rows in the README match the result files. `qos.csv` agent rewards equal `final_eval.csv` on all 100 rows. |
| 2 | Tie marking in the benchmark table | Yes. Recomputed with my own rule: 9 rows with a clear winner, 26 ties, no wrong mark; the README carries the same rows and a legend. |
| 3 | Unsourced NetworkGym reward claim | Yes. Removed from `docs/RESEARCHER_NOTE.md`, removal noted. (My script printed one "PROBLEM" for the string "log-ratio" in `docs/TESTER_REPORT.md`; that line describes the removal and asserts nothing. False positive.) |

Results unchanged: the re-run benchmark equals the previously committed one
on all 2,100 episodes with zero difference (checked against the file in git
history, not the Developer's copy). The only package code added since the
last confirmed commit is `carla_gym/utils/qos.py` (43 lines, reporting only).

README statements checked against the data: "54 to 69% dropped on the trace
scenarios" (data 54.4 to 69.2), PPO 92.2% delivered (92.18), cs448b reward
+0.41 with 57% dropped (+0.414, 57.5%).

One wording note, not a defect: delivered % plus dropped % exceeds 100 in 44%
of benchmark episodes, by up to 2.7 points, because a backlog left from the
warm-up drains inside the measured window. The tables say "the remainder is
backlog growth"; that growth can be negative. Worth half a sentence.

Regression: **124 passed** here and in a fresh clone; no tracked junk files.

### Tester/QA status after round 4

No open defects. The Researcher's observation that both trace scenarios drop
more than half their traffic is confirmed by my independent numbers; whether
to lower their target load is a design decision for the Developer and
Researcher, not a bug.

## 11. Confirmation pass on round 5: raw datasets removed (commit `e01aa70`)

Log: `qa/retest/confirm_round5.log`.

| Check | Result |
|---|---|
| Files under `data/` at `HEAD` | only `cs448b_trace.csv` and `pcap_trace.csv`; no raw file name anywhere in the tree |
| Local raw copies | 3 files in `data/raw/`, all ignored by git |
| Derived traces | byte-identical to before (same MD5 sums as at `12eb83c`) |
| Adapters, run on the local raw copies | reproduce both committed traces exactly |
| Address scan of all 172 tracked files, including the PNG images embedded in notebooks (byte scan) | no IPv4-shaped string |
| Tests with the raw files present | **129 passed** |
| Tests with `data/raw/` hidden (same as a fresh clone) | **115 passed, 14 skipped**; every skip carries the "raw datasets are not distributed" reason, none for any other reason |
| Results | README numbers, benchmark determinism and saved-model probes still pass |
| Notebooks | all four executed, in order, 0 errors, no stale references |

**Round 5 is confirmed for the current files.**

### One new finding

**R5-1 (LOW): a hardware-address fragment from the capture is still in the
code.** `carla_gym/core/dataset_adapters.py` line 22 gives, as an example of a
non-IP endpoint, a MAC-vendor string taken from the capture (vendor label
plus the last three octets of a network card's address). It is not an IP
address and identifies at most a virtual network card, but it is real data
from the removed file. Replace it with a made-up example. For the Developer.

### Limits of this check

* The scan reads bytes. Text drawn *inside* a chart image cannot be found
  that way. I reviewed notebook 03's code instead: every chart plots node
  ids, loads or degrees, never an address. The words "Source IP" and
  "Destination IP" in that notebook are column names in a description of
  supported input formats, not data.
* **History is not clean, as the Developer stated.** Two commits in the
  history touch the raw files, `origin/main` still contains all three
  (73 MB), and the old notebook 03 output with its address chart is in the
  history too. Removing the files from the current tree does not remove them
  from the public repository. That needs a history rewrite and a force-push,
  which is a decision for the user.

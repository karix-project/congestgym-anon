# Tester report: QA round 1 fixes

Developer hand-back for the Tester/QA role. Branch `fix/qa-round-1`, based on
`12eb83c` ("phase 2"). Not pushed. Work done 10 Oct 2026.

Every number below comes from a run on this machine; the log or CSV is named
next to it. Section 9 lists what was not verified.

## 1. Environment

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.14.5 (the only interpreter installed) |
| Key packages | gymnasium 1.4.0, numpy 2.5.3, pandas 3.0.6, networkx 3.7, torch 2.14.1, stable-baselines3 2.9.0, pytest 9.1.1, pettingzoo 1.27.0, matplotlib 3.11.2, nbconvert 7.17.1 |
| Full freeze | `qa/baseline/pip_freeze.txt` |

The virtual environment named in the handoff (`C:\Users\aryan\Downloads\carla\carla-gym`)
does not exist on this machine and the system Python had no packages. A new
`.venv` was created in the repo with the versions above (git-ignored).

## 2. Baseline before any change (`qa/baseline/`)

| Check | Result | Log |
|---|---|---|
| `pytest tests -q` | **34 passed, 1 failed** (handoff expected 35 passed) | `pytest.log` |
| Reference policies, unconstrained | max 0.464 / qp 0.256 / min -1.000 / random -1.190: matches the handoff | `random_agent_unconstrained.log` |
| Reference policies, budget 0.8 | qp 0.152 / max -0.027 / min -1.000 / random -1.190: matches the handoff | `random_agent_budget08.log` |
| Dataset conversion, 3 raw files | all convert; output byte-identical to the committed traces | `prepare_*.log` |
| Notebooks 01 to 04 as-is | all execute, 0 errors; 04's curve is flat (-0.196 to -0.192 over 60k steps) | `notebook_*.log` |
| PPO smoke run, 20k steps | `ep_rew_mean` flat at about -283; held-out PPO -1.000 +/- 0.000 | `ppo_smoke_20k.log` |

The failing test was `test_adapter_output_reproducible_across_processes`:
it spawned `python3`, which is not on PATH here.

## 3. Findings: reproduced, fixed, guarded

Repro scripts are in `qa/repro/`; before-logs in `qa/baseline/repro_*.log`,
after-logs in `qa/after/repro_*.log`.

| ID | Reproduced? | Fix (commit) | Regression tests | Before -> after |
|---|---|---|---|---|
| H1 seed replay | Yes, single- and multi-agent | `72cbe64`: `env.py`, `marl_env.py` re-seed traffic only on an explicit seed; `train_ppo.py` seeds each env rank | `test_unseeded_resets_draw_fresh_traffic`, `test_seeded_reset_replays_identical_traffic`, `test_constructor_seed_fixes_first_unseeded_episode`, `test_trace_traffic_unseeded_resets_differ`, `test_multi_agent_unseeded_resets_draw_fresh_traffic` | consecutive un-seeded episodes identical: True -> False |
| H2 corrupted README | Yes | `1a98925`: block repaired, `data/pcap_tr` removed | none (docs) | n/a |
| H3 stale README | Yes | `10bedcf` plus the results commit: full rewrite; result tables are copied from `results/` by `scripts/update_readme_results.py` | none (docs) | n/a |
| H4 PPO does not learn | Yes. Deterministic actions averaged 0.05; 100% of bandwidth entries below 0.4 (inside the M2 dead zone); policy std stayed 0.996 | `7e9da74` (action space rescaled to [-1, 1], `log_std_init=-1`), on top of H1, M2 | none (training outcome; see section 6) | held-out PPO -1.000 -> +0.494 +/- 0.013 |
| M1 reward blind spots | Yes. Constant weight 0.30: every buffer full, 0 deadline misses. (0.45 gave 17 misses here, not 0; raw ratio exceeded 1 on 44 of 120 steps, not 46. My scenario may differ from the QA one.) | `46fb356`: delay term includes age of queued packets; a packet that outlives its deadline in a queue is a miss, counted once | `test_packet_stuck_in_queue_counts_as_deadline_miss_once`, `test_starved_network_is_penalised_for_delay_and_deadlines`, `test_deadline_coefficient_is_configurable` | weight 0.30: 0 -> 1029 deadline misses, mean reward -0.915 -> -1.119 |
| M2 link-budget cliff | Yes. 12-unit packet at weight 0.3 delivered 0.0 units in 50 steps; constant weights up to 0.40 scored exactly -1.000 | `46fb356`: fractional link service | `test_packet_larger_than_link_budget_is_served_over_several_steps`, `test_zero_allocation_moves_nothing`, `test_link_never_serves_more_than_its_allocation`, `test_reward_increases_with_bandwidth_without_a_dead_zone` | delivered 0.0 -> 12.0; sweep now rises smoothly from -1.682 (0.1) to +0.550 (1.0) |
| **N1 buffer overflow (new)** | Found during this round. Original code held up to 988 units in a 120-unit buffer once traffic moved | `46fb356`: downstream space checked against start-of-step occupancy plus arrivals | `test_node_buffers_bounded_while_traffic_moves` (4 cases) | fullest node 988.3 -> 120.0 |
| M3 multi-agent | Yes. 8 of 9 star agents (1 to 3 on other topologies) had empty action spaces; PettingZoo undeclared; silent fallback | `eb1d1df`, `a660a6c`: balanced link ownership, nodes without links are not agents, ImportError without PettingZoo, dependency declared | `tests/test_marl_env.py` (15 tests) | empty-action agents: 8 -> 0 (star) |
| M4 render | Yes, `None` in both modes | `a3f6c58`: `rgb_array` returns an (H, W, 3) uint8 array | `test_render_rgb_array_returns_image`, `test_render_without_mode_returns_none`, `test_gymnasium_api_conformance_with_render` | NoneType -> ndarray (600, 750, 3) |
| M5 validation | Yes, all five cases (budget -0.5 gave applied sum -6.00; `gym.make(max_steps=500)` cut at 200) | `a3f6c58` | `test_invalid_bandwidth_budget_rejected`, `test_valid_bandwidth_budgets_run`, `test_wrong_length_action_gives_clear_error`, `test_grid_rejects_non_square_node_count`, `test_step_after_truncation_requires_reset`, `test_gym_make_honours_max_steps` | each case now raises a clear error or behaves correctly |
| M6 hygiene | Yes | `c80a0ff`, `84e3cec`, `a660a6c`: `.gitignore`; `.DS_Store` removed (`__pycache__` and `.idea` were NOT untracked by `a660a6c` as first stated here; the QA re-test caught it as T1 and it is fixed in round 2, see section 10); duplicate trace removed; versions pinned in `constraints.txt`; extras in `pyproject.toml`; `sys.executable` in the subprocess test; informal wording removed | `test_committed_traces_match_the_adapters` | n/a |
| LOW: trace mean load | Yes (pcap trace has 6 empty bins of 252) | `767fadf`: mean taken over all bins | `test_trace_mean_load_matches_target_including_empty_bins` | realised 106.1 (handoff figure) -> 110.6 in notebook 03 |
| LOW: unseeded random placement | Yes (by reading) | `767fadf`: `topology_seed` parameter and `--topology-seed` | `test_random_topology_placement_uses_the_env_seed` | n/a |
| LOW: `m` not exposed | Yes | `46fb356`: `CarlaGymEnv(m=...)` | `test_deadline_coefficient_is_configurable` | n/a |

Behaviour changes a reviewer should know about:

* The M1, M2 and N1 fixes change dynamics and reward. **No number produced
  before commit `46fb356` is comparable with numbers after it**, including
  the reference numbers in the handoff.
* `examples/random_agent_test.py` now correlates outcomes with the mean of
  the *bandwidth* half of the action. The old check used the mean of all 2E
  entries; routing weights have no monotone effect, and after the fixes that
  diluted correlation fell to 0.031 and failed the 0.05 threshold, while the
  bandwidth-only correlation is 0.158. This is a change to a check, made
  deliberately; please confirm you agree with it.
* `CarlaGym-v0` is registered without `max_episode_steps`; the env truncates
  itself at `max_steps`.
* `step()` after truncation now raises `RuntimeError`.
* `grid` with a non-square `num_nodes` raises instead of silently building 9 nodes.
* `data/pcap_trace_midterm.csv` was removed (byte-identical to
  `data/pcap_trace.csv`, regenerable). Raw datasets were not touched.

## 4. Additions for the abstract-compliance gaps

| Gap | Added | Where |
|---|---|---|
| Only one RL algorithm | SAC and TD3 next to PPO | `examples/train_agent.py` |
| No routing-aware baseline | `load_aware_routing`, `congestion_aware` | `carla_gym/baselines.py` |
| No scenario matrix | 5 topologies x 3 loads x 2 budgets, a bottleneck scenario, 2 traces x 2 budgets | `scripts/run_benchmark.py`, `results/benchmark/` |
| Bottleneck mitigation only implicit | `bottlenecks={node: scale}` | `core/network.py`, both envs |

### Abstract-compliance checklist

| Requirement | Evidence |
|---|---|
| Python RL framework on the Gymnasium API | `check_env` passes with and without the render check; `gym.make("CarlaGym-v0")` works |
| Dynamic packet routing | E routing weights in the action, Dijkstra per step; routing-aware baselines; PPO learns non-uniform routing weights |
| Bandwidth allocation | E bandwidth weights, optional budget; monotone-reward test |
| Node-bottleneck mitigation | `bottlenecks` option; benchmark row "grid/bottleneck node 4 x0.5": `congestion_aware` +0.003 vs `max_bandwidth` -0.196 |
| Customizable topologies | grid, ring, star, mesh, random; sizes, capacities, buffers configurable |
| Volatile data-load generation | Poisson QoS classes with `load_scale`; trace-driven traffic |
| Granular QoS reward (loss, latency) | four normalized terms, each reported in `info` |
| Train, evaluate, benchmark agents | `train_agent.py`, `summarize_training.py`, `run_benchmark.py` |
| PPO and a second agent vs rule-based baselines | section 6 |

## 5. Test inventory (final code)

`python -m pytest -q` -> **98 passed** (`qa/after/pytest.log`): `tests/test_env.py` 60,
`tests/test_dataset_pipeline.py` 23, `tests/test_marl_env.py` 15.

| Area | Checked by |
|---|---|
| Gymnasium API | `check_env` (2 tests), `gym.make` (2 tests) |
| PettingZoo API | `parallel_api_test` on grid, ring, star, mesh, random |
| Topologies | all five build and step |
| Budgets | None, 0.5, 0.8, 1.0 run; -0.5, 0, 1.5 rejected |
| Conservation | synthetic and trace traffic; also asserted during a 10-episode PPO rollout |
| Bounds | buffers (stalled, moving, double load, bottleneck), link service, observations |
| Seeding | seeded determinism, un-seeded independence, constructor seed, both envs, trace traffic |
| Datasets | all three raw files; schema, node range, no self-loops; committed traces equal adapter output; cross-process reproducibility under different `PYTHONHASHSEED` (test for the pcap file; manual run for cs448b and the Wireshark export, identical MD5s) |

Dataset counts: `cs448b_trace.csv` 4,644 rows, 92 steps (all with traffic);
`pcap_trace.csv` 4,163 rows, 252 steps (246 with traffic). The Wireshark
export drops 634 non-IP rows and yields the same trace.

Notebooks: all four executed in place with `jupyter nbconvert --execute`,
zero error outputs (`qa/after/notebook_*.log`).

Python 3.10 compatibility: every `.py` file parses with
`ast.parse(feature_version=(3, 10))`. Nothing was *run* on 3.10 to 3.13.

## 6. Training results

Scenario: default 3x3 grid, synthetic traffic, `bandwidth_budget=0.8`,
200-step episodes, reward averaged after a 20-step warm-up. Five training
seeds (0 to 4) per algorithm. Learning curves use validation seeds 1000 to
1004; the table uses test seeds 2000 to 2009. Files: `results/grid_b08/`.

| Policy | Kind | Reward (mean ± 95% CI) | n | CI over |
|---|---|---:|---:|---|
| PPO | RL agent | +0.494 ± 0.013 | 5 | training seeds |
| queue_proportional | baseline | +0.404 ± 0.024 | 10 | test episodes |
| congestion_aware | baseline | +0.386 ± 0.021 | 10 | test episodes |
| SAC | RL agent | +0.383 ± 0.028 | 5 | training seeds |
| load_aware_routing | baseline | +0.378 ± 0.026 | 10 | test episodes |
| max_bandwidth | baseline | +0.317 ± 0.030 | 10 | test episodes |
| random | baseline | -0.901 ± 0.026 | 10 | test episodes |
| min_bandwidth | baseline | -1.803 ± 0.008 | 10 | test episodes |

* **PPO** (500,000 steps): held-out reward rose from -0.487 at the first checkpoint
  (25,000 steps) to +0.460 at the last, averaged over seeds. On the test seeds
  it scores +0.494 +/- 0.013, above the best baseline (`queue_proportional`,
  +0.404 +/- 0.024); the intervals do not overlap.
* **SAC** (100,000 steps): held-out reward rose from -0.834 at 10,000 steps to
  +0.347 at 100,000. On the test seeds it scores +0.383 +/- 0.028 (per seed:
  +0.416, +0.395, +0.360, +0.365, +0.381). That is above `max_bandwidth` and far above `random`, and
  statistically indistinguishable from the three stronger heuristics. Its curve
  flattened after about 60,000 steps (+0.329 at 60,000, +0.347 at 100,000), so
  with these untuned settings SAC learns much faster per step than PPO but
  levels off lower. SAC had one fifth of PPO's environment steps; whether more
  steps or tuning would close the gap was not tested.
* Learning curves: `results/grid_b08/learning_curves.png`.

Checks on the PPO result:

* Per-seed test rewards: +0.496, +0.510, +0.493, +0.492, +0.481. All five are
  above the best baseline's mean.
* PPO (seed 0) vs `queue_proportional` on the test seeds: throughput 0.844 vs
  0.809, loss 0.078 vs 0.133, delay term 0.376 vs 0.372, deadline-miss 0.165
  vs 0.173. The gain is from delivering more and dropping less.
* Conservation and the buffer limit were asserted inside those rollouts.
* Before the fixes the same script gave PPO -1.000 +/- 0.000.

Baseline scenario matrix (`results/benchmark/`, 35 scenarios x 6 policies x
10 seeds = 2,100 episodes): best policy is `max_bandwidth` in 16 scenarios,
`queue_proportional` in 14, `congestion_aware` in 5. The full table is in
`results/benchmark/summary.md` and the README.

## 7. Not verified

* Python 3.10 to 3.13: not installed here. Syntax check only.
* Linux and macOS: not run. The bash commands in the README were not executed
  on a POSIX system (the repo's own scripts were run through Git Bash on Windows).
* `render_mode="human"`: runs under the Agg backend only (no window was shown).
* `pip install -e .` and the optional extras: not installed that way; the
  code was run from the repo root.
* TD3: a 6,000-step smoke run only. No multi-seed result.
* RL training on any scenario other than grid / budget 0.8, and on the traces
  beyond the short single-seed run in notebook 04.
* Multi-agent *training*: the env passes the API tests, but no MARL
  algorithm was trained on it.
* The baselines in `final_eval.csv` were computed at commit `b04db83`; later
  commits did not change default dynamics, and the benchmark run on the final
  code reproduces the same baseline means for this scenario (0.317, 0.404,
  0.378, 0.386). The RL agents were not retrained after `b04db83`.

## 8. Known limitations (documented in the README)

Slotted time, one hop per step; shared undirected link budget served in
node-id order; single FIFO per node; route fixed at admission; throughput
term clipped at 1; global observation in the multi-agent env.

## 9. What the Tester should re-check

1. Re-run `python -m pytest -q` in your own environment; expect 98 passed.
2. Re-run each `qa/repro/*.py` and compare with `qa/after/repro_*.log`.
3. H1: confirm with your own script, not mine, that SB3 training now sees
   different traffic per episode and per env.
4. N1 and M2 changed the simulator core. Please review `NetworkModel.forward`
   line by line: partial service (`pkt.sent`), the start-of-step occupancy
   rule, and that a dropped packet's link budget is still consumed.
5. Fractional service with a routing change: a packet part-way across a link
   keeps its path (routes are fixed at admission). Confirm no case exists
   where `pkt.sent` carries over to a different link.
6. M1: is counting a queued packet as a deadline miss when `age > deadline`
   (not `age + 1 > deadline`) the right boundary? It is the conservative one.
7. Deadline-miss denominator is delivered + expired packets in that step. With
   very few packets resolved the ratio is noisy. Judge whether that matters.
8. The changed sensitivity check in `random_agent_test.py` (section 3).
9. `step()` after truncation raising: check it does not break any wrapper you
   use (SB3 `DummyVecEnv` and Gymnasium `RescaleAction` were fine).
10. Multi-agent link ownership on `random` topologies with other seeds and
    sizes: every link owned once, no empty agent.
11. `bottlenecks` with several degraded nodes, and on a star hub.
12. `TraceTrafficGenerator` with `loop=False` past the end of the trace
    (returns no packets; reward terms then divide by zero offered load and
    fall back to 0). Not covered by a test.
13. Training reproducibility: rerun one PPO seed and see how close the test
    reward lands to the saved one. Bit-exact repeatability was not checked.
14. README commands in PowerShell from a fresh clone, including the install.
15. Notebook 04 part 2 trains live; its numbers will differ slightly between
    machines. Confirm the text it prints still matches what it plots.

## 10. Round 2: fixes after the QA re-test (`docs/QA_RETEST_REPORT.md`)

| ID | Finding | Fix | Guard |
|---|---|---|---|
| T1 | `__pycache__` (8 files) and `.idea` (6 files) were still tracked. I had staged their removal and then ran `git reset` to unstage unrelated files, which undid it; the commit message and this report then claimed something that had not happened. | `git rm -r --cached` on those paths; `git ls-files` now lists none. | none (repository state; check with `git ls-files`) |
| T2 | A NaN action was served as unlimited bandwidth (reward +0.768, 79.9 units through a 30-unit link). | `CarlaGymEnv.step` and `MultiAgentCarlaGymEnv.step` raise `ValueError` on NaN/inf; the multi-agent env also checks each agent's action shape; `NetworkModel.forward` and `update_routing` refuse non-finite, negative-routing or mis-shaped input. | `test_non_finite_action_rejected` (12 cases), `test_network_rejects_non_finite_allocation_and_routing`, `test_non_finite_agent_action_rejected` (2), `test_wrong_shape_agent_action_rejected` |

After round 2: `python -m pytest -q` -> **114 passed** (`tests/test_env.py` 73,
`tests/test_dataset_pipeline.py` 23, `tests/test_marl_env.py` 18). The counts
in sections 5 and 9 (98) describe the state before round 2.

Results are unaffected: with finite actions the simulator behaves exactly as
before (the tester's determinism probe still reproduces `results/benchmark`
to 1e-15 and the saved PPO/SAC models reproduce their saved rewards).

Not addressed in round 2, by instruction (only T1 and T2 were requested):
T3 to T10 in `docs/QA_RETEST_REPORT.md`.

## 11. Round 3: T3 to T10 from the QA re-test

| ID | Decision | What changed | Guard |
|---|---|---|---|
| T3 | Fixed | `RewardCalculator` rejects negative `k`, `c`, `m` and `delay_reference <= 0`; `TrafficGenerator` rejects negative `load_scale`. Zero is allowed for all four. | `test_invalid_reward_and_load_parameters_rejected` (6 cases), `test_zero_coefficients_and_zero_load_are_valid` |
| T4 | Fixed | `MultiAgentCarlaGymEnv.step` raises unless it gets exactly one action per live agent. | `test_step_requires_an_action_for_every_live_agent` |
| T5 | Documented, not changed | Deadline-term denominator can include a packet on two steps; expiry is recorded one step late. Described in `utils/reward.py` and the README limitations. | n/a |
| T6 | Documented | No backpressure; consequence spelled out in the README limitations. | n/a |
| T7 | Documented, not changed | Zero throughput credit in steps with no offered traffic. README limitations and `utils/reward.py`. | n/a |
| T8 | Fixed | "Sprint"/"Phase" wording removed from `examples/random_agent_test.py` and `tests/test_env.py` (the printed heading is now "Sanity checks"). | n/a |
| T9 | Documented, not changed | Pending-demand observation counts a part-sent packet at full size. README limitations. | n/a |
| T10 | Fixed | `scripts/summarize_training.py` and notebook 04 now average each baseline over all ten run directories instead of taking training-seed 0's run. Only the `random` row moves: -0.908 -> -0.901 (+/- 0.026). No other number changes. | Tester probe updated to the same method |

Why T5, T7 and T9 were documented rather than changed: each alters the reward
or the observation. Changing them would invalidate every trained model and
every table (about two hours of retraining and a full regeneration) for
effects the re-test measured as negligible at the loads used. If the
Researcher wants them changed, that is a deliberate re-run, not a patch.

After round 3: `python -m pytest -q` -> **122 passed** (`tests/test_env.py` 80,
`tests/test_dataset_pipeline.py` 23, `tests/test_marl_env.py` 19).
No dynamics, reward or observation code changed in rounds 2 and 3, so the
trained agents and the benchmark were not re-run.

## 12. Round 4: the three inconsistencies from `docs/RESEARCHER_REVIEW.md` (section 4)

| # | Inconsistency | What changed | Guard |
|---|---|---|---|
| 1 | Reported `throughput` / `loss` are means of per-step reward terms, not the fraction of traffic delivered / dropped | New `carla_gym/utils/qos.py` (`EpisodeQoS`) aggregates the raw counters. `scripts/run_benchmark.py` records delivered and dropped fraction, delay of delivered packets and deadline misses per delivered packet for every episode and writes `results/benchmark/qos.md`. New `scripts/evaluate_saved.py` produces the same for the saved agents and baselines (`results/grid_b08/qos.csv`, `qos.md`). `examples/random_agent_test.py` prints `delivered_frac` / `dropped_frac`. README and notebook 02 say which number is which. | `test_episode_qos_matches_raw_totals_and_differs_from_mean_of_step_ratios`, `test_episode_qos_empty_is_finite` |
| 2 | Benchmark table bolded a "best" policy in every row, including statistical ties | `summary.md` bolds a policy only when its 95% interval overlaps no other; tied leaders are marked `~`; a legend line explains both. README text says most rows are ties and that the far-below-zero rows are overloaded regimes. | n/a (presentation) |
| 3 | `docs/RESEARCHER_NOTE.md` said NetworkGym uses a log-ratio reward, without a source | Sentence removed and the removal noted in place. It had been copied from the handoff. | n/a |

Things to know:

* No reward, observation or dynamics code changed. The agents were not
  retrained. `evaluate_saved.py` re-evaluates the saved models; its rewards
  equal `results/grid_b08/summary.csv` exactly for PPO, SAC and the five
  deterministic baselines. Its `random` row differs slightly (-0.898 against
  -0.901) because it draws its own actions; the table says so.
* Trained agents in network terms (test seeds, steps 20 to 199): PPO delivers
  92.2 +/- 0.5 % and drops 7.7 +/- 0.5 %; `queue_proportional` 87.0 +/- 1.4 %
  and 13.0 +/- 1.4 %; SAC 87.6 +/- 0.9 % and 12.3 +/- 0.9 %. Delay of delivered
  packets: PPO 3.89, `queue_proportional` 3.88, SAC 4.30 steps.
* The aggregate view exposes how overloaded BOTH trace scenarios are: 54 to
  69% of offered traffic is dropped under every heuristic (pcap and cs448b,
  budgets 0.5 and 0.8), while the per-step loss term is far lower, because
  the traces are bursty and a per-step ratio weighs a quiet step as much as
  a heavy one. The cs448b scenario even has a positive reward (+0.41) with
  57% of traffic dropped. The README states this. It does not change any
  ranking, but anyone quoting "loss" for the trace scenarios must use the
  aggregate figure.
* The benchmark was re-run in full to add the new columns. All 2,100
  episodes reproduce the previously saved reward, throughput, loss, latency
  and deadline values with zero difference. The new table has 9 rows with a
  clear best policy and 26 rows marked as ties.

After round 4: `python -m pytest -q` -> **124 passed** (`tests/test_env.py` 82,
`tests/test_dataset_pipeline.py` 23, `tests/test_marl_env.py` 19).

Not addressed (out of scope for round 4): the experiments in section 5 of the
researcher review, the presentation claims in its section 6, and the user's
dataset decision.

## 13. Round 5: raw datasets removed from the repository (user decision)

Decision by the user (10 Oct 2026): remove the raw datasets, keep the derived
traces.

| What | Change |
|---|---|
| `data/cs448b_ipasn.csv`, `data/output1.csv`, `data/Midterm_53_group.csv` | Removed from the repository (untracked). Local copies were moved to `data/raw/`, which is git-ignored, so nothing was deleted from the user's machine. |
| `data/cs448b_trace.csv`, `data/pcap_trace.csv` | Kept. They contain node ids 0 to 15 only. |
| `tests/test_dataset_pipeline.py` | Adapter tests that need the raw files carry a `needs_raw` marker and are skipped when `data/raw/` is absent. The two env-level trace tests now run on the committed trace. Five new tests cover the committed traces without raw data (well-formed and anonymous, placement, drives the env). |
| `notebooks/03_dataset_exploration.ipynb` | Rewritten to use the derived traces only and re-executed (0 errors). The previous version's executed output contained a chart of real destination IP addresses. |
| README, `scripts/prepare_trace.py`, `dataset_adapters.py` docstring, notebook 02's assert message | State that the raw files are not distributed and how to build a trace from one's own data. |

Test counts: with the raw files present locally **129 passed**; without them
(a fresh clone) **115 passed, 14 skipped**. Both were run.

A scan of all tracked text files, notebooks and CSVs finds no IPv4 address.

**Not done, and it matters:** the raw files are still in the git history
(they were committed in `12eb83c` and earlier, and that history is already on
the public GitHub repository). So is the old notebook 03 output. Removing
them from history needs a history rewrite and a force-push of the public
repository, which changes every commit id and affects anyone who has cloned
it. I did not do that without explicit instruction. Until it is done, the
addresses remain publicly retrievable.

Also unchanged: the committed traces were produced from those raw files, so
their provenance and licence question (see `docs/RESEARCHER_REVIEW.md`,
section 7) still applies to the derived data, in a much weaker form.

## 14. Round 6: R5-1 from the round-5 confirmation

`carla_gym/core/dataset_adapters.py` quoted a real MAC-vendor string from the
removed capture as an example of a non-IP endpoint. It is replaced with a
made-up placeholder (`"Vendor_00:00:00"`). A search of all tracked files finds
no other MAC-style string and no remaining copy of the original in code,
docs or logs. Docstring only: no behaviour change, **129 passed**.

Like the raw files, the original string remains in the git history.

"""Tester/QA, 15 Oct 2026: checks on the 0.4.0 training matrix and on
docs/RL_RESULTS_0.4.0.md. Uses code paths separate from
qa/research/rl_results_analysis.py where possible.

    PYTHONPATH=. python qa/retest/qa_rl_results.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from scripts.run_training_matrix import ABLATIONS, SCENARIOS

R = "results/rl"
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


# ---------------------------------------------------------------- 1 completeness and shapes
# expected (obs dim, action dim) per scenario, from the env definitions:
# grid9 E=12 N=9 -> obs 33; grid16 E=24 N=16 -> obs 64; random9 (topology_seed 0) E=15 N=9 -> obs 39
SHAPES = {"grid_l125_b06": (33, 24), "grid_bottleneck_b08": (33, 24), "pcap40_grid16_b05": (64, 48),
          "random9_l15_b06": (39, 30), "grid_l125_b06_bandwidth_only": (33, 12),
          "grid_l125_b06_routing_only": (33, 12), "random9_l15_b06_routing_only": (39, 15)}
runs = sorted(glob.glob(os.path.join(R, "*", "*_seed*")))
check(len(runs) == 55, f"1 55 run directories ({len(runs)})")
missing, bad_shape, log_errors = [], [], []
for d in runs:
    name, run = d.split(os.sep)[-2], os.path.basename(d)
    for f in ("final_eval.csv", "eval_curve.csv", "model.zip", "train_monitor.csv", "train.log"):
        if not os.path.exists(os.path.join(d, f)):
            missing.append((d, f))
    log = open(os.path.join(d, "train.log"), encoding="utf-8", errors="replace").read()
    if re.search(r"Traceback|Error|nan", log):
        log_errors.append(d)
    algo = PPO if run.startswith("ppo") else SAC
    m = algo.load(os.path.join(d, "model.zip"), device="cpu")
    if (m.observation_space.shape[0], m.action_space.shape[0]) != SHAPES[name]:
        bad_shape.append((d, m.observation_space.shape, m.action_space.shape))
check(not missing, f"1 every run has its five output files {missing[:3]}")
check(not log_errors, f"1 no Traceback/Error/nan in any train.log {log_errors[:3]}")
check(not bad_shape, f"1 every model has the obs/action size of its scenario and ablation {bad_shape[:3]}")
n_ppo = sum(os.path.basename(d).startswith("ppo") for d in runs)
check(n_ppo == 35 and len(runs) - n_ppo == 20, f"1 35 PPO and 20 SAC runs ({n_ppo}, {len(runs)-n_ppo})")
ok_steps = all(pd.read_csv(os.path.join(d, "eval_curve.csv")).timesteps.max() ==
               (500000 if os.path.basename(d).startswith("ppo") else 100000) for d in runs)
check(ok_steps, "1 every run trained to its full budget (PPO 500k, SAC 100k)")

# ---------------------------------------------------------------- 2 headline env is the swept scenario
q = pd.read_csv(os.path.join(R, "grid_l125_b06", "qos.csv"))
test_ca = 100 * q[q.policy == "congestion_aware"].delivered_fraction.mean()
s30 = pd.read_csv("qa/research/headline_settle30_summary.csv")
val_ca = s30[s30.scenario == "grid9/load 1.25/budget 0.6"]["best_delivered_%"].iloc[0]
check(abs(test_ca - val_ca) < 2.0,
      f"2 headline congestion_aware delivered on test seeds ({test_ca:.1f}%) close to the 30-seed validation sweep ({val_ca}%)")
check(SCENARIOS["grid_l125_b06"] == ["--load-scale", "1.25", "--budget", "0.6"], "2 queue flags of the headline match the decision")

# ---------------------------------------------------------------- 3 seed separation
train = {s + r for s in range(5) for r in range(4)}          # seed + rank, n_envs <= 4
val, test = set(range(1000, 1005)), set(range(2000, 2010))
check(not (train & val) and not (train & test) and not (val & test), "3 training, validation and test seeds are disjoint")
check(set(q.test_seed) == test, "3 qos tables use test seeds 2000-2009 only")

# ---------------------------------------------------------------- 4 recompute from final_eval.csv
rng = np.random.default_rng(0)


def from_final(name, agent):
    rows = []
    for d in sorted(glob.glob(os.path.join(R, name, f"{agent.lower()}_seed*"))):
        f = pd.read_csv(os.path.join(d, "final_eval.csv"))
        best = f[f.policy.isin(HEUR)].groupby("policy").reward.mean().idxmax()
        a = f[f.policy == agent].set_index("test_seed").reward
        b = f[f.policy == best].set_index("test_seed").reward
        rows.append(a - b)
    diff = pd.concat(rows, axis=1)                       # test seeds x training seeds
    per_seed = diff.mean(axis=0).to_numpy()
    boot = [rng.choice(per_seed, len(per_seed)).mean() for _ in range(5000)]
    return per_seed.mean(), np.percentile(boot, [2.5, 97.5]), int((diff.values > 0).sum())


claims = {("grid_l125_b06", "PPO"): (-0.062, 0), ("grid_bottleneck_b08", "PPO"): (0.173, 50),
          ("pcap40_grid16_b05", "PPO"): (0.056, 50), ("random9_l15_b06", "PPO"): (0.025, 44)}
for (name, agent), (want, wins) in claims.items():
    m, (lo, hi), w = from_final(name, agent)
    check(abs(m - want) < 0.0005 and w == wins and (lo > 0 or hi < 0),
          f"4 {name} {agent} minus best heuristic reward {m:+.3f} (bootstrap 95% [{lo:+.3f}, {hi:+.3f}]), wins {w}/50; doc says {want:+.3f}, {wins}/50")

# ---------------------------------------------------------------- 5 audit the write-up against the log
log = open("qa/research/rl_results_analysis.log", encoding="utf-8").read()
doc = open("docs/RL_RESULTS_0.4.0.md", encoding="utf-8").read()
pairs = [("-0.62 +/- 0.56 pts (wins 16/50)", "-0.6 +/- 0.6 pts (wins 16/50)"),
         ("delivered +9.65 +/- 0.32 pts (wins 50/50)", "+9.7 +/- 0.3 pts"),
         ("delivered +3.95 +/- 0.68 pts (wins 50/50)", "+4.0 +/- 0.7 pts"),
         ("delivered +1.61 +/- 0.37 pts (wins 47/50)", "+1.6 +/- 0.4 pts"),
         ("reward +0.025 +/- 0.008 (wins 44/50)", "+0.025 +/- 0.008 (44/50)"),
         ("change 400k->500k +0.011 +/- 0.018", "+0.011 +/- 0.018"),
         ("change 400k->500k +0.032 +/- 0.026", "+0.032 +/- 0.026"),
         ("change 400k->500k +0.017 +/- 0.011", "+0.017 +/- 0.011"),
         ("bandwidth_only PPO, reward             -0.013 +/- 0.015", "-0.013 +/- 0.015"),
         ("routing_only PPO, reward             +0.038 +/- 0.020", "+0.038 +/- 0.020"),
         ("routing_only PPO, delivered_fraction +14.991 pts +/- 0.411", "+15.0 +/- 0.4 points"),
         ("tau +0.60", "headline 0.60")]
for in_log, in_doc in pairs:
    check(in_log in log and in_doc in doc, f"5 '{in_doc}' in the doc is backed by the log ('{in_log}')")
qmd = open(os.path.join(R, "grid_l125_b06", "qos.md"), encoding="utf-8").read()
check("| PPO | +0.037 | 73.5 ± 0.6 | 26.4 ± 0.5 | 5.82 ± 0.11 | 0.370 ± 0.006 |" in qmd and
      "| congestion_aware | +0.099 | 74.1 ± 1.4 | 25.8 ± 1.2 | 5.16 ± 0.09 | 0.330 ± 0.010 |" in qmd,
      "5 headline delay/deadline figures (5.82 vs 5.16, 0.370 vs 0.330) match qos.md")
r9 = open(os.path.join(R, "random9_l15_b06_routing_only", "qos.md"), encoding="utf-8").read()
check("| PPO | -0.060 | 66.3" in r9 and "| load_aware_routing | -0.130 | 64.8" in r9 and "| max_bandwidth | -0.175 | 63.3" in r9,
      "5 random9 routing-only figures (66.3 / 64.8 / 63.3) match qos.md")
old = open("docs/PUBLICATION_READINESS.md", encoding="utf-8").read()
check("+0.090 +/- 0.013" in old and "50 of 50" in old, "5 the quoted 0.3.0 result (+0.090, 50 of 50) matches its source")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL")

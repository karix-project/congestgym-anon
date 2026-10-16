"""Tester/QA, 16 Oct 2026: checks on experiments E2 and E3 (docs/E2_E3_RESULTS.md).

    PYTHONPATH=. python qa/retest/qa_e2_e3.py
"""
import glob
import os
import subprocess
from datetime import datetime

import pandas as pd
from stable_baselines3 import PPO

from carla_gym.utils.stats import ci95
from scripts.run_training_matrix import HP_SWEEP

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


SEL = open("results/rl_hp/SELECTED.txt", encoding="utf-8").readline().strip()
W = f"results/rl_hp/{SEL}"

# ---------------------------------------------------------------- 1 selection integrity
rows = []
for name in HP_SWEEP:
    v = [pd.read_csv(f).set_index("timesteps").reward_mean.loc[500000]
         for f in sorted(glob.glob(f"results/rl_hp/{name}/ppo_seed[0-2]/eval_curve.csv"))]
    rows.append((sum(v) / len(v), name, len(v)))
best = max(rows)[1]
check(best == SEL and all(n == 3 for _, _, n in rows), f"1 argmax of mean validation reward at 500k over seeds 0-2 is the recorded choice ({SEL})")
commit = subprocess.run(["git", "log", "-1", "--format=%ct", "--", "results/rl_hp/SELECTED.txt"],
                        capture_output=True, text=True).stdout.strip()
t_commit = int(commit)
later = [p for p in [f"{W}/ppo_seed3/final_eval.csv", f"{W}/ppo_seed4/final_eval.csv", f"{W}/qos.csv", f"{W}/summary.md"]
         if os.path.getmtime(p) > t_commit]
check(len(later) == 4, f"1 the choice was committed ({datetime.fromtimestamp(t_commit):%H:%M:%S}) before the extra seeds and the test tables were created")
others = [p for p in glob.glob("results/rl_hp/*/qos.csv") + glob.glob("results/rl_hp/*/summary.md") if not p.replace("\\", "/").startswith(W)]
check(not others, f"1 no other sweep configuration was evaluated or summarised {others}")

# ---------------------------------------------------------------- 2 hyperparameters and shapes inside the models
ok = True
for s in range(5):
    m = PPO.load(f"{W}/ppo_seed{s}/model.zip", device="cpu")
    ok &= (m.gamma, m.learning_rate, m.ent_coef) == (0.99, 3e-4, 0.01)
check(ok, "2 all 5 winner models carry gamma 0.99, lr 3e-4, ent_coef 0.01")
m = PPO.load("results/rl_hp/g0.95_lr1e-4_ent0/ppo_seed0/model.zip", device="cpu")
check((m.gamma, m.learning_rate, m.ent_coef) == (0.95, 1e-4, 0.0), "2 a different sweep configuration carries its own values (g0.95, lr 1e-4, ent 0)")
for folder, act in [("grid_bottleneck_b08_bandwidth_only", 12), ("grid_bottleneck_b08_routing_only", 12),
                    ("random9_l15_b06_bandwidth_only", 15)]:
    shapes = {PPO.load(f, device="cpu").action_space.shape[0] for f in glob.glob(f"results/rl/{folder}/ppo_seed*/model.zip")}
    n = len(glob.glob(f"results/rl/{folder}/ppo_seed*/final_eval.csv"))
    check(shapes == {act} and n == 5, f"2 {folder}: 5 runs, action size {act} (half of the full action)")


# ---------------------------------------------------------------- 3 recompute from final_eval.csv
def final(folder, agent="PPO"):
    out = []
    for d in sorted(glob.glob(f"{folder}/ppo_seed*")):
        f = pd.read_csv(f"{d}/final_eval.csv")
        out.append(f[f.policy == agent].set_index("test_seed").reward.rename(int(d[-1])))
    return pd.concat(out, axis=1).T                                   # train seed x test seed


def base(folder, policy):
    f = pd.read_csv(sorted(glob.glob(f"{folder}/ppo_seed*/final_eval.csv"))[0])
    return f[f.policy == policy].set_index("test_seed").reward


tp = final(W)
d = tp.sub(base("results/rl/grid_l125_b06", "congestion_aware"), axis=1)
per = d.mean(axis=1)
got = f"{per.mean():+.3f} +/- {ci95(per.to_numpy()):.3f}"
check(got == "-0.043 +/- 0.015" and int((d.values > 0).sum()) == 1 and per.mean() + ci95(per.to_numpy()) < 0,
      f"3 E2 pre-registered figure recomputed from final_eval.csv: {got}, wins {(d.values > 0).sum()}/50; CI entirely below 0")
cw = base(W, "congestion_aware")
check((cw - base("results/rl/grid_l125_b06", "congestion_aware")).abs().max() == 0,
      "3 the sweep's own final_eval baseline rows equal the headline folder's (same env, same test seeds)")
claims = {("grid_bottleneck_b08", "bandwidth_only"): "+0.424 +/- 0.042", ("grid_bottleneck_b08", "routing_only"): "+0.037 +/- 0.013",
          ("random9_l15_b06", "bandwidth_only"): "-0.000 +/- 0.006", ("random9_l15_b06", "routing_only"): "+0.413 +/- 0.011",
          ("grid_l125_b06", "bandwidth_only"): "-0.013 +/- 0.015", ("grid_l125_b06", "routing_only"): "+0.038 +/- 0.020"}
for (full, half), want in claims.items():
    dd = final(f"results/rl/{full}") - final(f"results/rl/{full}_{half}")
    pp = dd.mean(axis=1)
    got = f"{pp.mean():+.3f} +/- {ci95(pp.to_numpy()):.3f}"
    check(got == want, f"3 E3 {full} full minus {half}: {got} (doc {want})")

# ---------------------------------------------------------------- 4 audit of the write-up
doc = open("docs/E2_E3_RESULTS.md", encoding="utf-8").read()
log = open("qa/research/e2_e3_analysis.log", encoding="utf-8").read()
for frag_doc, frag_log in [("-0.043 +/- 0.015, tuned PPO ahead in 1 of 50", "reward: -0.043 +/- 0.015 (wins 1/50)"),
                           ("-0.037 +/- 0.015", "minus tuned congestion_aware   reward -0.037 +/- 0.015"),
                           ("| Reward | +0.037 | +0.056 | +0.099 |", "tuned 0.056, original 0.037"),
                           ("| Delay of delivered packets (steps) | 5.82 | 5.58 | 5.16 |", "tuned PPO 5.58, original PPO 5.82, congestion_aware 5.16"),
                           ("| Deadline misses per delivered packet | 0.370 | 0.357 | 0.330 |", "tuned PPO 0.357, original PPO 0.370, congestion_aware 0.330"),
                           ("-0.6 +/- 0.7", "-0.61 +/- 0.73 pts"),
                           ("93.8%", "delivered 93.8%"), ("72.8%", "delivered 72.8%"),
                           ("+0.5 +/- 0.3 points", "+0.458 pts +/- 0.286")]:
    check(frag_doc in doc and frag_log in log, f"4 doc '{frag_doc}' backed by log '{frag_log}'")
check("1646836" in doc and subprocess.run(["git", "cat-file", "-t", "1646836"], capture_output=True, text=True).stdout.strip() == "commit",
      "4 the selection commit quoted in the doc exists")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL")

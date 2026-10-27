"""Researcher, 15 Oct 2026: first reading of the 0.4.0 training matrix
(medium budget: PPO 500k steps, SAC 100k, 5 training seeds, test seeds
2000-2009). Reads results/rl/*/qos.csv (from scripts/evaluate_saved.py) and
results/rl/*/*/eval_curve.csv.

    PYTHONPATH=. python qa/research/rl_results_analysis.py

1. Agent vs best heuristic, paired on test seeds. The unit is the training
   seed: for each, the mean over the 10 test seeds of (agent - heuristic);
   the CI is over the 5 training seeds. Also the count of (training seed,
   test seed) pairs the agent wins, out of 50.
2. Convergence: validation reward at the last checkpoints, mean over seeds,
   and the change over the last fifth of training.
3. Ablations on the headline and on random9: full-control PPO against
   one-half PPO, paired on test seeds and training seeds.
4. Whether reward and delivered traffic rank the policies the same way.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from congestgym.utils.stats import ci95  # noqa: E402

R = "results/rl"
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
SCEN = ["grid_l125_b06", "grid_bottleneck_b08", "pcap40_grid16_b05", "random9_l15_b06"]


def qos(name):
    return pd.read_csv(os.path.join(R, name, "qos.csv"))


def paired_vs(q, agent, other, col):
    """Per-training-seed mean of (agent - other) over test seeds; `other` is a
    baseline (one row per test seed) or another agent table (train x test)."""
    a = q[q.policy == agent].pivot(index="train_seed", columns="test_seed", values=col)
    if isinstance(other, str):
        b = q[q.policy == other].set_index("test_seed")[col]
        diff = a.sub(b, axis=1)
    else:
        diff = a - other
    per_seed = diff.mean(axis=1)
    wins = int((diff.values > 0).sum())
    return per_seed.mean(), ci95(per_seed.to_numpy()), wins, diff.size


print("=" * 100)
print("1. Agent minus best heuristic, paired on test seeds; CI over 5 training seeds")
print("=" * 100)
for name in SCEN:
    q = qos(name)
    best = q[q.policy.isin(HEUR)].groupby("policy").reward.mean().idxmax()
    best_d = q[q.policy.isin(HEUR)].groupby("policy").delivered_fraction.mean().idxmax()
    print(f"\n{name}   (best heuristic by reward: {best}; by delivered: {best_d})")
    for agent in ["PPO", "SAC"]:
        m, h, w, n = paired_vs(q, agent, best, "reward")
        md, hd, wd, _ = paired_vs(q, agent, best_d, "delivered_fraction")
        print(f"   {agent}: reward {m:+.3f} +/- {h:.3f} (wins {w}/{n})   "
              f"delivered {100*md:+.2f} +/- {100*hd:.2f} pts (wins {wd}/{n})")

print("\n" + "=" * 100)
print("2. Convergence: validation reward (seeds 1000-1004), mean over training seeds")
print("=" * 100)
for name in SCEN + ["grid_l125_b06_bandwidth_only", "grid_l125_b06_routing_only", "random9_l15_b06_routing_only"]:
    for algo in ["ppo", "sac"]:
        runs = sorted(glob.glob(os.path.join(R, name, f"{algo}_seed*", "eval_curve.csv")))
        if not runs:
            continue
        c = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(runs)])
        piv = c.pivot(index="timesteps", columns="seed", values="reward_mean")
        last = piv.index.max()
        fifth = piv.index[piv.index <= last * 0.8].max()
        change = piv.loc[last] - piv.loc[fifth]
        tail = " ".join(f"{piv.loc[t].mean():+.3f}" for t in piv.index[-3:])
        print(f"   {name:30s} {algo.upper()}  last 3 checkpoints {tail}   "
              f"change {fifth//1000}k->{last//1000}k {change.mean():+.3f} +/- {ci95(change.to_numpy()):.3f}")

print("\n" + "=" * 100)
print("3. Ablations: full-control PPO minus one-half PPO, paired on (training seed, test seed)")
print("=" * 100)
for full, halves in [("grid_l125_b06", ["grid_l125_b06_bandwidth_only", "grid_l125_b06_routing_only"]),
                     ("random9_l15_b06", ["random9_l15_b06_routing_only"])]:
    qf = qos(full)
    for half in halves:
        qh = qos(half)
        for col, scale, unit in [("reward", 1, ""), ("delivered_fraction", 100, " pts")]:
            a = qf[qf.policy == "PPO"].pivot(index="train_seed", columns="test_seed", values=col)
            b = qh[qh.policy == "PPO"].pivot(index="train_seed", columns="test_seed", values=col)
            d = (a - b).mean(axis=1)
            print(f"   {full} PPO minus {half.replace(full + '_', '')} PPO, {col:18s} "
                  f"{scale*d.mean():+.3f}{unit} +/- {scale*ci95(d.to_numpy()):.3f}")
    for half in halves:
        qh = qos(half)
        hp = qh[qh.policy == "PPO"].groupby("train_seed")[["reward", "delivered_fraction"]].mean()
        print(f"   {half:30s} PPO: reward {hp.reward.mean():+.3f}, delivered {100*hp.delivered_fraction.mean():.1f}%")
    print(f"   {full:30s} PPO: reward {qf[qf.policy=='PPO'].reward.mean():+.3f}, "
          f"delivered {100*qf[qf.policy=='PPO'].delivered_fraction.mean():.1f}%")

print("\n" + "=" * 100)
print("4. Reward vs delivered ranking of all policies except random/min (Kendall tau)")
print("=" * 100)
for name in SCEN:
    q = qos(name)
    m = q[~q.policy.isin(["random", "min_bandwidth"])].groupby("policy")[["reward", "delivered_fraction"]].mean()
    r, d = m.reward.to_numpy(), m.delivered_fraction.to_numpy()
    n = len(r); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(r[i] - r[j]) * np.sign(d[i] - d[j])
    order_r = " > ".join(m.reward.sort_values(ascending=False).index)
    print(f"   {name:22s} tau {s/(n*(n-1)/2):+.2f}   by reward: {order_r}")

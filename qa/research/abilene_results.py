"""Researcher, 18 Oct 2026: reading of the Abilene scenario (abilene_l120_b08).

1. PPO and SAC minus the best rule-based policy, paired on test seeds (CI
   over the 5 training seeds; wins out of 50 (training seed, test seed)
   pairs), reward and delivered. Paired, because the time of day each episode
   samples moves every policy together (docs/ABILENE_SCENARIO.md).
2. Convergence: validation reward over the last fifth of training.
3. Detours: share of delivered packets on a longer-than-shortest path
   (hop count), as in qa/research/e4_detours.py.

    PYTHONPATH=. python qa/research/abilene_results.py
"""
import glob
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from congestgym.baselines import POLICIES, policy_rng            # noqa: E402
from congestgym.utils.stats import ci95                          # noqa: E402
from examples.train_agent import make_env, parse_args           # noqa: E402
from scripts.run_training_matrix import SCENARIOS               # noqa: E402

NAME = "abilene_l120_b08"
R = os.path.join("results", "rl", NAME)
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
q = pd.read_csv(os.path.join(R, "qos.csv"))

print("1. Agent minus best rule-based policy, paired on test seeds 2000-2009; CI over 5 training seeds")
for col, scale, unit in [("reward", 1, ""), ("delivered_fraction", 100, " pts")]:
    best = q[q.policy.isin(HEUR)].groupby("policy")[col].mean().idxmax()
    b = q[q.policy == best].set_index("test_seed")[col]
    for agent in ["PPO", "SAC"]:
        a = q[q.policy == agent].pivot(index="train_seed", columns="test_seed", values=col)
        d = a.sub(b, axis=1)
        per = d.mean(axis=1).to_numpy()
        print(f"   {agent} minus {best:18s} {col:18s} {scale*per.mean():+.3f}{unit} +/- {scale*ci95(per):.3f}"
              f"   (wins {(d.values > 0).sum()}/{d.size})")
    second = q[q.policy.isin(HEUR)].groupby("policy")[col].mean().sort_values(ascending=False).index[1]
    d2 = (q[q.policy == best].set_index("test_seed")[col] - q[q.policy == second].set_index("test_seed")[col]).to_numpy()
    print(f"   (best minus second-best rule-based, {best} - {second}: {scale*d2.mean():+.3f}{unit} +/- {scale*ci95(d2):.3f})")

print("\n2. Convergence: validation reward (seeds 1000-1004), mean over training seeds")
for algo in ["ppo", "sac"]:
    runs = sorted(glob.glob(os.path.join(R, f"{algo}_seed*", "eval_curve.csv")))
    c = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(runs)])
    piv = c.pivot(index="timesteps", columns="seed", values="reward_mean")
    last = piv.index.max(); fifth = piv.index[piv.index <= last * 0.8].max()
    ch = piv.loc[last] - piv.loc[fifth]
    print(f"   {algo.upper()}: last 3 checkpoints {' '.join(f'{piv.loc[t].mean():+.3f}' for t in piv.index[-3:])}; "
          f"change {fifth//1000}k->{last//1000}k {ch.mean():+.3f} +/- {ci95(ch.to_numpy()):.3f}")

print("\n3. Detours (delivered packets born at step >= 20, test seeds 2000-2009)")
args = parse_args(SCENARIOS[NAME])


def episode(act_fn, seed, rescale):
    env = make_env(args, 0, rescale=rescale)()
    base = env.unwrapped
    hop = dict(nx.all_pairs_shortest_path_length(base.network.graph))
    born = []
    gen = base.traffic.generate
    base.traffic.generate = lambda t, _g=gen: (lambda p: (born.extend(p), p)[1])(_g(t))
    obs, _ = env.reset(seed=seed)
    for _ in range(args.steps):
        obs, *_ = env.step(act_fn(base, obs))
    st = np.array([(len(p.path) - 1) / hop[p.src][p.dst] for p in born if p.birth_step >= 20 and p.path and p.arrived])
    return st.mean(), np.mean(st > 1 + 1e-9)


for name in HEUR:
    rows = [episode(lambda e, o, fn=POLICIES[name], rng=policy_rng(s): fn(e, o, rng), s, False) for s in range(2000, 2010)]
    rows = np.array(rows)
    print(f"   {name:20s} stretch {rows[:, 0].mean():.4f}   on a detour {100*rows[:, 1].mean():.2f}% +/- {100*ci95(rows[:, 1]):.2f}")
for algo, cls in [("ppo", PPO), ("sac", SAC)]:
    per = []
    for path in sorted(glob.glob(os.path.join(R, f"{algo}_seed*", "model.zip"))):
        m = cls.load(path, device="cpu")
        per.append(np.mean([episode(lambda e, o, m=m: m.predict(o, deterministic=True)[0], s, True)
                            for s in range(2000, 2010)], axis=0))
    per = np.array(per)
    print(f"   {algo.upper():20s} stretch {per[:, 0].mean():.4f}   on a detour {100*per[:, 1].mean():.2f}% +/- {100*ci95(per[:, 1]):.2f}  (5 seeds)")

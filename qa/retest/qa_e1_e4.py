"""Tester/QA, 15 Oct 2026: checks on experiments E1 and E4 (docs/E1_E4_RESULTS.md).

    PYTHONPATH=. python qa/retest/qa_e1_e4.py
"""
import os
import re

import networkx as nx
import numpy as np
import pandas as pd

from carla_gym.baselines import make_policy, policy_rng
from carla_gym.utils.stats import ci95
from examples.train_agent import make_env, parse_args
from scripts.evaluate_saved import episode
from scripts.run_training_matrix import SCENARIOS, known_configurations

H = "qa/research"
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


val = pd.read_csv(f"{H}/e1_validation_episodes.csv", float_precision="round_trip")
test = pd.read_csv(f"{H}/e1_test_episodes.csv", float_precision="round_trip")

# ---------------------------------------------------------------- 1 seed discipline
check(set(val.seed) == set(range(1000, 1010)), "1 selection used validation seeds 1000-1009 only")
check(set(test.seed) == set(range(2000, 2010)), "1 test episodes use test seeds 2000-2009 only")
n_settings = val.groupby(["scenario", "policy"]).params.nunique()
check(all(n_settings.xs("congestion_aware", level=1) == 24) and all(n_settings.xs("queue_proportional", level=1) == 4),
      "1 grid sizes as planned (24 and 4 settings per scenario)")
check(test.groupby(["scenario", "policy"]).params.nunique().max() == 1, "1 exactly one chosen setting per scenario and policy reached the test seeds")
check(all("floor=0.05 base=0.1 slope=0.9 signal=max" in set(val[(val.scenario == s) & (val.policy == "congestion_aware")].params)
          for s in SCENARIOS), "1 the default setting is in the congestion_aware grid for every scenario")

# ---------------------------------------------------------------- 2 E1's code path reproduces the published evaluation
worst = 0.0
for s in SCENARIOS:
    q = pd.read_csv(f"results/rl/{s}/qos.csv", float_precision="round_trip")
    targs = parse_args(SCENARIOS[s])
    for pol in ("congestion_aware", "queue_proportional"):
        fn = make_policy(pol)
        for seed in (2000, 2005):
            rng = policy_rng(seed)
            r = episode(targs, lambda e, o: fn(e, o, rng), seed, False, 20)
            want = q[(q.policy == pol) & (q.test_seed == seed)].reward.iloc[0]
            worst = max(worst, abs(r["reward"] - want))
check(worst == 0.0, f"2 default settings through E1's code path equal results/rl/*/qos.csv exactly (16 episodes, max diff {worst:.1e})")

# ---------------------------------------------------------------- 3 recompute choices and effects
log = open(f"{H}/e1_tune_heuristics.log", encoding="utf-8").read()
for (s, pol), g in val.groupby(["scenario", "policy"]):
    best = g.groupby("params").reward.mean().idxmax()
    chosen = test[(test.scenario == s) & (test.policy == pol)].params.iloc[0]
    check(best == chosen and f"{s:22s} {pol:20s} chosen [{best}]" in log, f"3 {s} {pol}: argmax of validation reward is the setting tested [{best}]")
doc = open("docs/E1_E4_RESULTS.md", encoding="utf-8").read()
for s, want_r in [("grid_l125_b06", "-0.006 +/- 0.011"), ("grid_bottleneck_b08", "+0.031 +/- 0.011"),
                  ("pcap40_grid16_b05", "+0.109 +/- 0.011"), ("random9_l15_b06", "+0.057 +/- 0.014")]:
    q = pd.read_csv(f"results/rl/{s}/qos.csv", float_precision="round_trip")
    t = test[(test.scenario == s) & (test.policy == "congestion_aware")].set_index("seed").reward
    u = q[q.policy == "congestion_aware"].set_index("test_seed").reward
    d = (t - u).to_numpy()
    got = f"{d.mean():+.3f} +/- {ci95(d):.3f}"
    check(got == want_r and want_r in doc, f"3 {s}: tuned minus untuned congestion_aware reward recomputed {got}, doc {want_r}")

# PPO vs strongest, recomputed by a different route: pooled over all 50 pairs, then per-seed CI
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
claims = {"grid_l125_b06": ("-0.062", "-0.6"), "grid_bottleneck_b08": ("+0.172", "+9.7"),
          "pcap40_grid16_b05": ("+0.044", "+2.3"), "random9_l15_b06": ("+0.025", "+1.6")}
for s, (cr, cd) in claims.items():
    q = pd.read_csv(f"results/rl/{s}/qos.csv", float_precision="round_trip")
    pools = {p: q[q.policy == p].groupby("test_seed")[["reward", "delivered_fraction"]].mean() for p in HEUR}
    for p in ("congestion_aware", "queue_proportional"):
        t = test[(test.scenario == s) & (test.policy == p)].groupby("seed")[["reward", "delivered"]].mean()
        pools["tuned " + p] = t.rename(columns={"delivered": "delivered_fraction"})
    ppo = q[q.policy == "PPO"]
    out = []
    for col in ("reward", "delivered_fraction"):
        best = max(pools, key=lambda k: pools[k][col].mean())
        m = ppo.merge(pools[best][col].rename("b"), left_on="test_seed", right_index=True)
        out.append((m[col] - m.b).mean() * (1 if col == "reward" else 100))
    check(f"{out[0]:+.3f}" == cr and f"{out[1]:+.1f}" == cd, f"3 {s}: PPO minus strongest recomputed {out[0]:+.3f} / {out[1]:+.1f} pts; doc {cr} / {cd}")

# ---------------------------------------------------------------- 4 E4 sanity on raw packets
def packets(scen, pol, seed):
    args = parse_args(known_configurations()[scen])
    env = make_env(args, 0, rescale=False)()
    base = env.unwrapped
    born = []
    gen = base.traffic.generate
    base.traffic.generate = lambda t, _g=gen: (lambda p: (born.extend(p), p)[1])(_g(t))
    obs, _ = env.reset(seed=seed)
    fn = make_policy(pol); rng = policy_rng(seed)
    for _ in range(args.steps):
        obs, *_ = env.step(fn(base, obs, rng))
    return base.network.graph, [p for p in born if p.birth_step >= 20 and p.path and p.arrived]


bad_path = below_one = 0
for scen in ("grid_bottleneck_b08", "random9_l15_b06"):
    for pol in ("max_bandwidth", "queue_proportional", "congestion_aware"):
        g, pk = packets(scen, pol, 2003)
        for p in pk:
            if any(not g.has_edge(u, v) for u, v in zip(p.path, p.path[1:])) or len(set(p.path)) != len(p.path):
                bad_path += 1
            if (len(p.path) - 1) < nx.shortest_path_length(g, p.src, p.dst):
                below_one += 1
        if pol in ("max_bandwidth", "queue_proportional"):
            det = sum((len(p.path) - 1) > nx.shortest_path_length(g, p.src, p.dst) for p in pk)
            check(det == 0, f"4 {scen} {pol}: no detours with uniform routing weights ({len(pk)} packets)")
check(bad_path == 0 and below_one == 0, "4 every delivered path is a simple walk on the graph and never shorter than the shortest path")

e4 = open(f"{H}/e4_detours.log", encoding="utf-8").read()
for frag in ["load_aware_routing (over test seeds) hops 1.598", "on a detour 6.29% +/- 1.03",
             "congestion_aware (over test seeds) hops 1.676", "on a detour 7.84% +/- 0.34",
             "PPO full control (5 seeds)         hops 1.624", "on a detour 0.43% +/- 0.25",
             "on a detour 1.13% +/- 0.33", "on a detour 0.61% +/- 0.20", "on a detour 0.34% +/- 0.12", "on a detour 0.76% +/- 0.31"]:
    check(frag in e4, f"5 E4 log contains '{frag}'")
for frag in ["6.29 +/- 1.03%", "7.84 +/- 0.34%", "0.43 +/- 0.25%", "1.13 +/- 0.33%", "0.61 +/- 0.20%", "0.34 +/- 0.12%", "0.76 +/- 0.31%"]:
    check(frag in doc, f"5 doc quotes '{frag}' from the E4 log")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL")

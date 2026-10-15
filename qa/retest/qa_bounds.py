"""Tester/QA, 15 Oct 2026: carla_gym/bounds.py and RL_RESULTS_0.4.0.md section 2.6.

    PYTHONPATH=. python qa/retest/qa_bounds.py
"""
import itertools
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from carla_gym.bounds import expected_demand, flow_bound, scenario_bound
from examples.train_agent import make_env, parse_args
from scripts.run_training_matrix import SCENARIOS

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


env = make_env(parse_args(SCENARIOS["grid_l125_b06"]), 0, rescale=False)().unwrapped
net, d = env.network, expected_demand(env.traffic, env.num_nodes)
lp = flow_bound(net, d, env.bandwidth_budget, "shortest")["delivered"]

# ---------------------------------------------------------------- 1 analytic knapsack bound
# Only the total budget: sum over links of load/capacity <= budget * E. A unit of
# pair (s,t) delivered on a shortest path uses hops(s,t)/capacity of allocation.
# Serving the shortest pairs first maximises delivered units (fractional knapsack).
cap = float(net.get_edge_capacities()[0])
hops = dict(nx.all_pairs_shortest_path_length(net.graph))
items = sorted((hops[s][t], d[s, t]) for s in range(9) for t in range(9) if s != t)
room = env.bandwidth_budget * net.num_edges() * cap          # capacity-hops per step
knap = 0.0
for h, dem in items:
    take = min(dem, room / h)
    knap += take; room -= take * h
    if room <= 1e-12:
        break
check(lp <= knap + 1e-6, f"1 LP ({lp:.3f}) never exceeds the analytic budget-only knapsack bound ({knap:.3f})")
check(abs(lp - knap) < 1e-6, f"1 LP equals the knapsack bound on the headline (per-link limits do not bind): diff {lp-knap:+.2e}")

# ---------------------------------------------------------------- 2 path-based formulation
paths = {(s, t): list(nx.all_shortest_paths(net.graph, s, t)) for s in range(9) for t in range(9) if s != t}
cols = [(st, p) for st, ps in paths.items() for p in ps]
E = net.num_edges()
eidx = {net._key(u, v): k for k, (u, v) in enumerate(net.edge_list())}
nv = E + len(cols)
A, b = [], []
for k in range(E):                                           # link: sum of path flows over it <= cap * a_k
    row = np.zeros(nv); row[k] = -cap
    for j, (_, p) in enumerate(cols):
        if any(eidx[net._key(u, v)] == k for u, v in zip(p, p[1:])):
            row[E + j] = 1.0
    A.append(row); b.append(0.0)
row = np.zeros(nv); row[:E] = 1.0; A.append(row); b.append(env.bandwidth_budget * E)
for st in paths:                                             # per pair: total path flow <= demand
    row = np.zeros(nv)
    for j, (st2, _) in enumerate(cols):
        if st2 == st:
            row[E + j] = 1.0
    A.append(row); b.append(d[st])
c = np.zeros(nv); c[E:] = -1.0
res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, 1)] * E + [(0, None)] * len(cols), method="highs")
check(res.status == 0 and abs(-res.fun - lp) < 1e-6,
      f"2 path-based LP ({len(cols)} shortest paths enumerated) gives the same bound: {-res.fun:.4f} vs {lp:.4f}")

# ---------------------------------------------------------------- 3 trace demand
penv = make_env(parse_args(SCENARIOS["pcap40_grid16_b05"]), 0, rescale=False)().unwrapped
pd_ = expected_demand(penv.traffic, penv.num_nodes)
check(abs(pd_.sum() - 40.0) < 1e-6 and np.all(np.diag(pd_) == 0), f"3 pcap expected demand sums to the target load 40 ({pd_.sum():.6f})")

# ---------------------------------------------------------------- 4 free == shortest, everywhere
tr = pd.read_csv("results/bounds/training.csv"); bm = pd.read_csv("results/bounds/benchmark.csv")
both = pd.concat([tr[["scenario", "bound_free", "bound_shortest"]], bm[["scenario", "bound_free", "bound_shortest"]]])
diff = (both.bound_free - both.bound_shortest).abs().max()
check(len(both) == 35 and diff < 1e-6, f"4 free and shortest-path bounds equal in all {len(both)} scenarios (max diff {diff:.1e})")
# Is it because no link is at its per-link limit? Count LP links at a_e = 1 per scenario.
at_cap = []
for name, flags in SCENARIOS.items():
    e = make_env(parse_args(flags), 0, rescale=False)()
    at_cap.append((name, int((scenario_bound(e, "free")["allocation"] > 1 - 1e-6).sum())))
print("   links at allocation 1.0 in the free-routing LP:", at_cap)

# ---------------------------------------------------------------- 5 exceedances only where bound is 100%
rows = [(r.scenario, r.bound_free, r.best_heuristic_delivered) for r in bm.itertuples()]
rows += [(r.scenario + " " + who, r.bound_free, getattr(r, col)) for r in tr.itertuples()
         for who, col in (("heuristic", "best_heuristic_delivered"), ("PPO", "ppo_delivered"), ("SAC", "sac_delivered"))]
exceed = [(s, b_, v) for s, b_, v in rows if v > b_ + 1e-9]
worst = max((v - b_ for _, b_, v in exceed), default=0.0)
check(all(b_ > 0.9999 for _, b_, _ in exceed) and worst <= 0.003 + 1e-9,
      f"5 every exceedance ({len(exceed)}) is in a scenario whose bound is 100%; largest {100*worst:.2f} points")

# ---------------------------------------------------------------- 6 audit of section 2.6
doc = open("docs/RL_RESULTS_0.4.0.md", encoding="utf-8").read()
log = open("qa/research/lp_static_policy.log", encoding="utf-8").read()
for scen, bound, scaled in [("grid_l125_b06", "90.3", "66.8 +/- 1.5"), ("grid_bottleneck_b08", "100.0", "62.8 +/- 0.8"),
                            ("pcap40_grid16_b05", "100.0", "60.7 +/- 1.5"), ("random9_l15_b06", "99.4", "71.1 +/- 1.6")]:
    line = [x for x in log.splitlines() if x.startswith(scen)][0]
    check(bound in line and line.count(scaled.split()[0]) >= 1 and scaled in doc,
          f"6 {scen}: bound {bound} and scaled fixed policy {scaled} match the log and the doc")
check("than 0.3\npoints" in doc.replace("\r\n", "\n") or "more than 0.3 points" in doc, "6 doc states the 0.3-point exceedance")
check("equal split (68.5%)" in doc, "6 equal split 68.5% on the headline (results/rl/grid_l125_b06/qos.md)")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL")

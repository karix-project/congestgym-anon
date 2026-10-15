"""Researcher, 15 Oct 2026: how loose is the fluid flow bound?

Takes the bound's own optimal link allocation (carla_gym/bounds.py, routing
"shortest") and plays it as a fixed policy in the simulator: that bandwidth
allocation every step, uniform routing weights (hop-count shortest paths,
random ties). Delivered fraction on test seeds 2000-2009, steps 20..199, as
in the result tables. These are achievable policies, so they give lower
bounds on the optimum, the flow bound an upper one.

Two variants:
  LP-static      the LP's allocation as returned (a vertex: just enough for
                 the mean load, no slack, spare budget left unused);
  LP-scaled      the same proportions scaled up to spend the whole budget,
                 each link capped at 1 (water-filling the spare budget).

    PYTHONPATH=. python qa/research/lp_static_policy.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.bounds import scenario_bound              # noqa: E402
from carla_gym.utils.qos import EpisodeQoS               # noqa: E402
from carla_gym.utils.stats import ci95                   # noqa: E402
from examples.train_agent import make_env, parse_args    # noqa: E402
from scripts.run_training_matrix import SCENARIOS        # noqa: E402

HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
print("delivered % on test seeds 2000-2009 (steps 20..199), mean +/- 95% CI over test seeds")
print(f"{'scenario':22s} {'flow bound':>10s} {'LP-static':>16s} {'LP-scaled':>16s} {'best heuristic':>26s} {'PPO':>6s}")
for name, flags in SCENARIOS.items():
    args = parse_args(flags)
    env0 = make_env(args, 0, rescale=False)()
    b = scenario_bound(env0, "shortest")
    alloc = np.clip(b["allocation"], 0.0, 1.0)
    budget_total = env0.unwrapped.bandwidth_budget * len(alloc)
    scaled = alloc.copy()
    for _ in range(50):                       # water-fill: scale, cap at 1, repeat on the rest
        free = scaled < 1.0 - 1e-12
        spare = budget_total - scaled.sum()
        if spare <= 1e-9 or not free.any() or scaled[free].sum() <= 0:
            break
        scaled[free] *= 1.0 + spare / scaled[free].sum()
        scaled = np.minimum(scaled, 1.0)

    def run(a):
        action = np.concatenate([a, np.ones_like(a)]).astype(np.float32)
        out = []
        for s in range(2000, 2010):
            env = make_env(args, 0, rescale=False)()
            env.reset(seed=s)
            qos = EpisodeQoS()
            for t in range(args.steps):
                _, _, _, _, info = env.step(action)
                if t >= 20:
                    qos.add(info)
            out.append(qos.summary()["delivered_fraction"])
        return out

    vals, vals2 = run(alloc), run(scaled)
    q = pd.read_csv(os.path.join("results", "rl", name, "qos.csv"))
    m = q.groupby("policy").delivered_fraction.mean()
    best = m[HEUR].idxmax()
    print(f"{name:22s} {100*b['delivered_fraction']:10.1f} {100*np.mean(vals):10.1f} +/- {100*ci95(vals):.1f}"
          f" {100*np.mean(vals2):10.1f} +/- {100*ci95(vals2):.1f}"
          f" {best:>19s} {100*m[best]:.1f} {100*m['PPO']:6.1f}"
          f"   (LP allocation: {np.count_nonzero(alloc > 1e-6)}/{len(alloc)} links used, sum {alloc.sum():.2f} of {budget_total:.2f}; scaled sum {scaled.sum():.2f})")

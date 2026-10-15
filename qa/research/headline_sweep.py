"""Researcher, 15 Oct 2026: choose the headline training scenario for 0.4.0.

Runs on VALIDATION seeds (1000-1009) so the choice does not use the test
seeds (2000+) on which results will be reported.

Criteria, fixed before running:
  C1 congested but workable: best heuristic delivers 65-88% of traffic
  C2 bandwidth matters: max over routing settings of |delivered(bandwidth-
     aware) - delivered(equal split)| >= 3 points
  C3 routing matters: max over bandwidth settings of |delivered(load-aware
     routing) - delivered(hop-count)| >= 3 points
  C4 reward and delivered agree on the ranking of the four heuristics
     (Kendall tau >= 0.67)
  C5 stable: 95% CI half-width of the best heuristic's reward <= 0.03
Ties go to the smaller (cheaper to train) scenario.

    PYTHONPATH=. python qa/research/headline_sweep.py
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.baselines import POLICIES          # noqa: E402
from carla_gym.utils.stats import ci95            # noqa: E402
from scripts.run_benchmark import run_episode     # noqa: E402

SEEDS = range(1000, 1010)
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
POLS = ["random"] + HEUR


def cells():
    out = []
    for load, b in itertools.product([1.0, 1.25, 1.5], [0.5, 0.6, 0.7]):
        out.append(dict(name=f"grid9/load {load:g}/budget {b:g}", topology="grid",
                        num_nodes=9, load_scale=load, budget=b))
    for load, b in itertools.product([1.0, 1.25], [0.6, 0.8]):
        out.append(dict(name=f"grid9 bottleneck 4x0.5/load {load:g}/budget {b:g}", topology="grid",
                        num_nodes=9, load_scale=load, budget=b, bottlenecks={4: 0.5}))
    for load, b in itertools.product([1.5, 2.0], [0.5, 0.7]):
        out.append(dict(name=f"grid16/load {load:g}/budget {b:g}", topology="grid",
                        num_nodes=16, load_scale=load, budget=b))
    return out


def job(args):
    sc, pol, seed = args
    r = run_episode(sc, POLICIES[pol], seed, 200, 20)
    return {"scenario": sc["name"], "policy": pol, "seed": seed,
            "reward": r["reward"], "delivered": r["delivered_fraction"]}


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def main():
    jobs = [(sc, p, s) for sc in cells() for p in POLS for s in SEEDS]
    with Pool(10) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=4))
    out = os.path.dirname(__file__)
    df.to_csv(os.path.join(out, "headline_sweep_episodes.csv"), index=False)

    rows = []
    for name, g in df.groupby("scenario", sort=False):
        m = g.groupby("policy")[["reward", "delivered"]].mean()
        d = 100 * m["delivered"]
        best = m.loc[HEUR, "reward"].idxmax()
        ci = ci95(g[g.policy == best]["reward"].to_numpy())
        bw = max(abs(d["queue_proportional"] - d["max_bandwidth"]),
                 abs(d["congestion_aware"] - d["load_aware_routing"]))
        rt = max(abs(d["load_aware_routing"] - d["max_bandwidth"]),
                 abs(d["congestion_aware"] - d["queue_proportional"]))
        tau = kendall(m.loc[HEUR, "reward"].to_numpy(), m.loc[HEUR, "delivered"].to_numpy())
        best_d = d[HEUR].max()
        crit = [65 <= best_d <= 88, bw >= 3, rt >= 3, tau >= 0.67, ci <= 0.03]
        rows.append({"scenario": name, "best_heuristic": best,
                     "best_reward": round(m.loc[best, "reward"], 3), "ci95": round(ci, 3),
                     "best_delivered_%": round(best_d, 1),
                     "random_delivered_%": round(d["random"], 1),
                     "bandwidth_effect_pts": round(bw, 1), "routing_effect_pts": round(rt, 1),
                     "kendall_tau": round(tau, 2),
                     "criteria_met": "".join(str(i + 1) if c else "." for i, c in enumerate(crit)),
                     "n_met": sum(crit)})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, "headline_sweep_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print("\nper-policy delivered % (validation seeds):")
    print((100 * df.pivot_table(index="scenario", columns="policy", values="delivered", sort=False))
          [POLS].round(1).to_string())


if __name__ == "__main__":
    main()

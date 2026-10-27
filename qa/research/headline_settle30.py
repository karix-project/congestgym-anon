"""Researcher, 15 Oct 2026: settle the headline scenario on 30 validation seeds.

The 10-seed sweeps put the stability criterion within noise: which plain-grid
cell passed C5 flipped with the RNG fix (docs/HEADLINE_SCENARIO.md). This
reruns the nine plain 3x3 grid cells on validation seeds 1000-1029. Test
seeds (2000+) stay unused.

Rules, fixed before running. The two documented criteria defects are corrected:
  C1 best heuristic delivers 65-88%
  C2 bandwidth helps: max(queue_prop - max_bw, congestion - load_aware) >= 3 points (signed)
  C3 routing helps:   max(load_aware - max_bw, congestion - queue_prop) >= 3 points (signed)
  C4 Kendall tau between reward and delivered ranking of the 4 heuristics >= 2/3
  C5 95% CI half-width of the best heuristic's reward <= 0.03
  C6 congestion_aware beats load_aware_routing and queue_proportional
     (paired over seeds, 95% CI lower bound > 0)
  Tie-break among cells meeting all six: the largest weaker C6 lower bound,
  then the lower load.

    PYTHONPATH=. python qa/research/headline_settle30.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from congestgym.utils.stats import ci95                 # noqa: E402
from headline_sweep import HEUR, POLS, cells, job, kendall  # noqa: E402

SEEDS = range(1000, 1030)


def paired(piv, a, b):
    d = (piv[a] - piv[b]).to_numpy()
    return d.mean(), ci95(d)


def main():
    grid = [c for c in cells("headline") if c["name"].startswith("grid9/")]
    jobs = [(sc, p, s) for sc in grid for p in POLS for s in SEEDS]
    with Pool(10) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=4))
    here = os.path.dirname(__file__)
    df.to_csv(os.path.join(here, "headline_settle30_episodes.csv"), index=False)

    rows = []
    for name, g in df.groupby("scenario", sort=False):
        m = g.groupby("policy")[["reward", "delivered"]].mean()
        piv = g.pivot(index="seed", columns="policy", values="delivered") * 100
        best = m.loc[HEUR, "reward"].idxmax()
        ci = ci95(g[g.policy == best]["reward"].to_numpy())
        best_d = 100 * m.loc[HEUR, "delivered"].max()
        bw = max(paired(piv, "queue_proportional", "max_bandwidth")[0],
                 paired(piv, "congestion_aware", "load_aware_routing")[0])
        rt = max(paired(piv, "load_aware_routing", "max_bandwidth")[0],
                 paired(piv, "congestion_aware", "queue_proportional")[0])
        tau = kendall(m.loc[HEUR, "reward"].to_numpy(), m.loc[HEUR, "delivered"].to_numpy())
        c_la, h_la = paired(piv, "congestion_aware", "load_aware_routing")
        c_qp, h_qp = paired(piv, "congestion_aware", "queue_proportional")
        weaker_lb = min(c_la - h_la, c_qp - h_qp)
        crit = [65 <= best_d <= 88, bw >= 3, rt >= 3, tau >= 2 / 3 - 1e-9, ci <= 0.03, weaker_lb > 0]
        rows.append({"scenario": name, "load": float(name.split("load ")[1].split("/")[0]),
                     "best_heuristic": best, "best_reward": round(m.loc[best, "reward"], 3),
                     "ci95": round(ci, 4), "best_delivered_%": round(best_d, 1),
                     "bandwidth_pts": round(bw, 2), "routing_pts": round(rt, 2), "tau": round(tau, 2),
                     "both_vs_routing_only": f"{c_la:+.2f} +/- {h_la:.2f}",
                     "both_vs_bandwidth_only": f"{c_qp:+.2f} +/- {h_qp:.2f}",
                     "weaker_C6_lower_bound": round(weaker_lb, 2),
                     "criteria_met": "".join(str(i + 1) if c else "." for i, c in enumerate(crit)),
                     "all_met": all(crit)})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(here, "headline_settle30_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"validation seeds {SEEDS.start}-{SEEDS.stop - 1}, {len(SEEDS)} per cell\n")
    print(s.drop(columns=["load", "all_met"]).to_string(index=False))
    q = s[s.all_met].sort_values(["weaker_C6_lower_bound", "load"], ascending=[False, True])
    print("\nqualifying (all six), in tie-break order:")
    print(q[["scenario", "weaker_C6_lower_bound", "best_delivered_%", "ci95"]].to_string(index=False)
          if len(q) else "(none)")
    if len(q):
        print(f"\nSELECTED: {q.iloc[0].scenario}")


if __name__ == "__main__":
    main()
